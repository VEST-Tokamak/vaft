"""Magnetic diagnostics: from a coil voltage to a current, a flux, or a mode number.

Three groups, with different provenance and different applicability.

**Sensor geometry** is pure arithmetic on a layout: where the array's centre
is and what poloidal angle each sensor sits at.

**The legacy VEST equilibrium chain** is the ported workflow that turns raw
magnetic diagnostic records into the currents and fluxes an equilibrium
reconstruction consumes.  Its functions are named ``vest_*``, are
VEST-specific, and carry shot-era acquisition policy: which samples the
output window covers, and which samples the baseline is fitted over, both
depend on the shot number.  Those policies live in ``vest.yaml`` and reach
these functions through a configuration record.

**Mode analysis** conditions Mirnov coil signals and extracts a toroidal mode
number, either from the cross-spectral phase of a pair or from a wrapped-phase
fit across an array.

Notation
--------
V        : coil terminal voltage                                     [V]
Phi      : poloidal flux through a loop                             [Wb]
B_p      : poloidal field at a probe                                 [T]
I_p      : plasma current                                            [A]
n        : toroidal mode number                                      [-]
f        : frequency                                                [Hz]

Conventions
-----------
**Integration carries Faraday's minus sign, applied before the baseline is
removed.**  A flux loop's flux is the negative time integral of its voltage,
and a poloidal probe's field likewise; the constant of integration is absorbed
into the fitted baseline rather than chosen separately.  Flux is stored per
radian, which the ODS mapper multiplies back by two pi.

**The two generic integrating routines hedge their sign and the VEST ones
state it.**  ``flux_loop_flux`` and ``b_field_pol_probe_field`` say the minus
applies "if that is the convention", because they take whatever a caller hands
them; the ``vest_*`` functions know their own diagnostics and assert it.  That
difference is deliberate and is preserved rather than harmonized.

**Plasma-current polarity is inferred from the data, not asserted.**
:func:`rogowski_coil_ip` flips the sign when the negative excursion dominates,
which is the one place in the module where a convention is decided by the
signal rather than declared.

**Toroidal mode numbers adhere to standard right-handed cylindrical coordinates.**
Coordinates (R, phi, Z) are oriented such that phi increases counter-clockwise
when viewed from above. A positive toroidal mode number (n > 0) denotes a
perturbation propagating in the positive phi direction (co-current / toroidal
direction). Across toroidally separated sensors, the signal phase varies as
theta(phi) = theta_0 - n * phi (negative phase slope d(theta)/d(phi) = -n). For
two coils separated by toroidal angle Delta_phi = phase_geometry > 0, the
cross-spectral phase of the downstream coil relative to the upstream coil is
Delta_theta = -n * Delta_phi, so the mode number is n = -Delta_theta / Delta_phi.
Both :func:`toroidal_mode_analysis` and :func:`toroidal_phase_fit_at_time`
share this convention.

Provenance
----------
.. [1] The legacy VEST magnetics workflow in MATLAB, principally
   ``VEST_MagneticSignalProcessing.m`` and its native-DAQ successor, from
   which the ``vest_*`` chain is ported with its window and baseline indices
   preserved.
.. [2] ``vest.yaml`` ``magnetics``, which carries the shot-era window and
   baseline policy these functions take as configuration (issue #195).
.. [3] The legacy ``vest_osc`` and ``vest_mirnov`` routines, whose
   preprocessing and spectrogram normalization the mode-analysis group
   reproduces.
"""

import warnings
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy import signal
from scipy.signal import coherence, csd, find_peaks, savgol_filter

from vaft.compat import cumtrapz_compat
from vaft.formula.statistics import rms
from .signal_processing import define_baseline, subtract_baseline

# ``vaft.database`` is imported inside the one function that needs it, not
# here: vaft.database.raw imports Matplotlib at module scope as an
# availability probe, and nothing about processing a waveform needs it.  Every
# consumer of vaft.process used to pay for it (issue #249; the Matplotlib
# chain is the one described in #268).  The ipywidgets slider closures that
# also lived here were retired with issue #485 -- reading a processed signal
# is what vaft.plot is for.

__all__ = [
    "DEFAULT_VEST_MAGNETICS_PROCESSING",
    "DegenerateBaselineWindowError",
    "MAGNETICS_CLUSTER_GAP_FRACTION",
    "MirnovSpectrogramResult",
    "ToroidalModeResult",
    "ToroidalPhaseFitResult",
    "ToroidalPhaseModeFit",
    "UnsupportedMagneticsDaqModeError",
    "VestEquilibriumMagneticsResult",
    "VestMagneticsProcessingConfig",
    "b_field_pol_probe_field",
    "flux_loop_flux",
    "magnetics_sensor_centre",
    "magnetics_sensor_poloidal_angle",
    "mirnov_preprocess_signal",
    "mirnov_spectrogram",
    "rogowski_coil_ip",
    "toroidal_mode_analysis",
    "toroidal_phase_fit_at_time",
    "vest_b_field_pol_probe_legacy",
    "vest_equilibrium_magnetics_detailed",
    "vest_equilibrium_magnetics_signals",
    "vest_flux_loop_flux_from_voltage",
    "vest_flux_loop_legacy",
    "vest_flux_loop_voltage",
    "vest_magnetics_time_window",
    "vest_md_signals",
]

# Naming convention for function name: {diagnostics_name}_{processing_quantity}

#: A sensor layout counts as two clusters (an inboard and an outboard array)
#: when the widest gap in R spans at least this fraction of the radial extent;
#: a ring of sensors round the plasma has gaps far smaller than that.
MAGNETICS_CLUSTER_GAP_FRACTION = 0.4


def magnetics_sensor_centre(r, z):
    """Geometric centre of a magnetic sensor layout.

    Parameters
    ----------
    r : array_like
        Major radius of each sensor [m].
    z : array_like
        Height of each sensor [m].

    Returns
    -------
    tuple of float
        The centre as major radius and height [m].

    Convention
    ----------
    A layout counts as **two clusters** -- an inboard and an outboard array --
    when the widest gap in major radius spans at least a set fraction of the
    radial extent; a ring of sensors around the plasma has gaps far smaller than
    that. The two cases need different centres: a ring's centre is the median of
    its sensors, while two clusters take the midpoint between them, because the
    median of a two-cluster layout lands inside whichever cluster has more
    sensors rather than between them.

    The median is used rather than the mean so that an unevenly populated ring
    does not drag the centre toward its dense side.

    Defaults
    --------
    The cluster gap fraction is a numerical convenience separating the two
    layouts; a real ring and a real two-cluster array sit far either side of it.

    Applicability
    -------------
    Machine-independent.  The VEST case is an example: an inboard array against a
    wall array further out is the two-cluster geometry this distinguishes.

    Limitations
    -----------
    Two cases only. Three or more clusters, or a layout that is neither, gets
    whichever branch its widest gap selects.

    Provenance
    ----------
    .. [1] Issue #486, which established this definition of the layout centre.
    """
    from vaft.plot.selection import classify_regions, radial_divider

    r = np.asarray(r, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()
    finite = np.isfinite(r) & np.isfinite(z)
    if not np.any(finite):
        raise ValueError("a sensor centre needs at least one finite (r, z) position")
    r, z = r[finite], z[finite]
    r0 = float(np.mean(r))
    split = radial_divider(r)
    extent = float(r.max() - r.min())
    if split and extent > 0.0:
        regions = np.asarray(classify_regions(r, split=split))
        inboard = r[regions == "inboard"]
        outboard = r[regions == "outboard"]
        if inboard.size and outboard.size:
            gap = float(outboard.min() - inboard.max())
            if gap >= MAGNETICS_CLUSTER_GAP_FRACTION * extent:
                r0 = 0.5 * (float(np.median(inboard)) + float(np.median(outboard)))
    return r0, float(np.median(z))


def magnetics_sensor_poloidal_angle(r, z, centre=None):
    """Poloidal angle of each sensor about the layout centre.

    Parameters
    ----------
    r : array_like
        Major radius of each sensor [m].
    z : array_like
        Height of each sensor [m].
    centre : tuple of float, optional
        The centre to measure about; computed from the layout when omitted [m].

    Returns
    -------
    np.ndarray
        Poloidal angle of each sensor, on the interval from zero to a full turn
        [rad].

    Convention
    ----------
    Measured from the layout centre, counter-clockwise from the outboard
    midplane, and wrapped to a single positive turn rather than centred on zero.

    **This is not the IMAS poloidal angle of a probe**, which describes the
    orientation of the probe's sensitive axis rather than its position; on VEST
    every vertical-field probe has the same IMAS angle wherever it sits. Confusing
    the two puts every probe at the same place.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Inherits the two-case centre rule of :func:`magnetics_sensor_centre` when the
    centre is not supplied, so a layout that fits neither case gets angles about a
    centre that may not be where the caller expects.

    Provenance
    ----------
    .. [1] The IMAS data dictionary's own poloidal-angle definition, which this
       deliberately differs from.
    """
    r = np.asarray(r, dtype=float)
    z = np.asarray(z, dtype=float)
    if centre is None:
        centre = magnetics_sensor_centre(r, z)
    r0, z0 = float(centre[0]), float(centre[1])
    angle = np.mod(np.arctan2(z - z0, r - r0), 2.0 * np.pi)
    # A sensor on the outboard midplane a rounding error below Z = 0 wraps to
    # 2*pi; it belongs at 0.
    return np.where(np.isclose(angle, 2.0 * np.pi), 0.0, angle)


class UnsupportedMagneticsDaqModeError(NotImplementedError):
    """Raised for a magnetics acquisition era with no ported processing path."""


class DegenerateBaselineWindowError(ValueError):
    """Raised when a baseline window contains fewer than two valid samples."""


@dataclass(frozen=True)
class VestMagneticsProcessingConfig:
    """Default VEST magnetics processing settings used by legacy `vfit_equilibrium_magnetics`.

    The values intentionally preserve the long-running VEST EFIT input workflow
    while making the knobs explicit for reproducibility and parameter scans.
    Shots 41446--41451 and shots from 41660 onward use indices 6500--9000
    with a 5000-sample probe baseline. Other shots use indices 6000--8500
    with an 8500-sample probe baseline. These are legacy acquisition-era
    policies, not automatic signal-quality decisions.
    """

    time_start: float = 0.0
    time_end: float = 0.99996
    sample_count: int = 25_000
    fast_sample_rate: float = 250_000.0
    lowpass_cutoff: float = 2_500.0
    lowpass_taps: int = 251
    default_index_start: int = 6000
    default_index_end: int = 8500
    default_probe_baseline_end: int = 8500
    late_shot_min: int = 41660
    transient_shot_min: int = 41446
    transient_shot_max: int = 41451
    late_index_start: int = 6500
    late_index_end: int = 9000
    late_probe_baseline_end: int = 5000
    flux_baseline_first_start: int = 3499
    flux_baseline_first_end: int = 5000
    flux_baseline_second_start: int = 11999
    flux_baseline_second_end: int = 15000
    flux_baseline_late_start: int = 5999
    flux_baseline_late_end: int = 7000
    flux_baseline_late_loop_numbers: tuple[int, ...] = (9, 10, 11)
    calibration_mode: str = "divide"
    flux_output_per_radian: bool = True
    # Shot-era policy resolved from `vest.yaml` (issue #195). When
    # `window_override` is None the legacy hardcoded thresholds below still
    # apply, so directly-constructed configs keep their historical behavior.
    window_override: tuple[int, int, int] | None = None
    flux_baseline_window: tuple[float, float] | None = None
    flux_baseline_samples: int | None = None
    daq_mode: str = "legacy"
    allow_zero_fallback: bool = False

    def timebase(self) -> np.ndarray:
        return np.linspace(self.time_start, self.time_end, self.sample_count)

    def window_for_shot(self, shot: int) -> tuple[int, int, int]:
        if self.window_override is not None:
            return self.window_override
        if self.transient_shot_min <= shot <= self.transient_shot_max or shot >= self.late_shot_min:
            return self.late_index_start, self.late_index_end, self.late_probe_baseline_end
        return self.default_index_start, self.default_index_end, self.default_probe_baseline_end


DEFAULT_VEST_MAGNETICS_PROCESSING = VestMagneticsProcessingConfig()


@dataclass(frozen=True)
class MirnovSpectrogramResult:
    """Time-frequency result for one Mirnov waveform."""

    time: np.ndarray
    frequency: np.ndarray
    magnitude: np.ndarray


@dataclass(frozen=True)
class ToroidalModeResult:
    """Cross-phase toroidal mode-number result."""

    frequency: np.ndarray
    n: np.ndarray
    power: np.ndarray
    phase: np.ndarray
    spectrum_frequency: np.ndarray
    cross_power: np.ndarray
    peak_indices: np.ndarray
    n_raw: np.ndarray
    n_rounded: np.ndarray
    coherence: np.ndarray


@dataclass(frozen=True)
class ToroidalPhaseModeFit:
    """Wrapped toroidal phase fit for one fluctuation frequency."""

    frequency: float
    n: int
    intercept: float
    rms_error: float
    phase: np.ndarray
    fitted_phase: np.ndarray
    amplitude: np.ndarray


@dataclass(frozen=True)
class ToroidalPhaseFitResult:
    """Mode-line fits from a selected time slice."""

    time: float
    toroidal_angle: np.ndarray
    modes: tuple[ToroidalPhaseModeFit, ...]
    candidate_n: np.ndarray


def _calibrated_signal(values: np.ndarray, calibration: float, mode: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if mode == "divide":
        return values / calibration
    if mode == "multiply":
        return values * calibration
    raise ValueError(f"Unsupported VEST magnetics calibration mode: {mode}")


def _linear_baseline(
    time_axis: np.ndarray,
    values: np.ndarray,
    indices: np.ndarray,
    *,
    allow_zero_fallback: bool = False,
    shot: int | None = None,
    channel: str | int | None = None,
    window_desc: str | None = None,
) -> np.ndarray:
    indices = np.asarray(indices, dtype=int)
    in_bounds = indices[(indices >= 0) & (indices < values.size)]
    valid = in_bounds[np.isfinite(values[in_bounds]) & np.isfinite(time_axis[in_bounds])]
    if valid.size < 2:
        context_parts = []
        if shot is not None:
            context_parts.append(f"shot {shot}")
        if channel is not None:
            context_parts.append(f"channel {channel!r}")
        context_str = f" [{', '.join(context_parts)}]" if context_parts else ""

        if allow_zero_fallback:
            warnings.warn(
                f"Degenerate baseline window{context_str}: {valid.size} valid sample(s) available "
                f"(requires >= 2). Falling back to zero baseline because allow_zero_fallback=True.",
                UserWarning,
                stacklevel=2,
            )
            return np.zeros(values.size, dtype=float)

        prefix = f"for {', '.join(context_parts)}: " if context_parts else ""
        time_span = (
            f"[{time_axis[0]:.6g}, {time_axis[-1]:.6g}] s"
            if time_axis.size > 0
            else "[]"
        )
        req_desc = f" ({window_desc})" if window_desc else ""
        if indices.size > 0:
            index_range = f"[{indices.min()}, {indices.max()}] (count: {indices.size})"
        else:
            index_range = "[] (empty)"

        raise DegenerateBaselineWindowError(
            f"Degenerate baseline window {prefix}found {valid.size} valid sample(s) (requires >= 2). "
            f"Requested indices{req_desc}: {index_range}. "
            f"Signal sample count: {values.size}, time span: {time_span}."
        )

    return np.polyval(np.polyfit(time_axis[valid], values[valid], 1), time_axis)


def _validate_daq_mode(cfg: VestMagneticsProcessingConfig) -> None:
    """Reject configs whose `daq_mode` disagrees with the rules it carries.

    `daq_mode` must not be decorative: the acquisition era it names and the
    flux-loop baseline rule that actually distinguishes the two donor
    functions have to agree, or a config could claim `native_daq` while
    silently processing flux loops the legacy way.
    """
    if cfg.daq_mode not in {"legacy", "native_daq"}:
        raise UnsupportedMagneticsDaqModeError(
            f"Unknown VEST magnetics daq_mode {cfg.daq_mode!r}; expected 'legacy' or 'native_daq'"
        )
    if cfg.daq_mode == "native_daq" and cfg.flux_baseline_samples is None:
        raise UnsupportedMagneticsDaqModeError(
            "daq_mode='native_daq' requires flux_baseline_samples: on the native "
            "acquisition the flux loops take the probes' leading-sample baseline "
            "(the donor reuses index_Bz_start:index_Bz_end). Without it the flux "
            "loops would silently fall back to the legacy baseline rule."
        )
    if cfg.daq_mode == "legacy" and cfg.flux_baseline_samples is not None:
        raise UnsupportedMagneticsDaqModeError(
            "flux_baseline_samples is a native-DAQ rule but daq_mode='legacy'; "
            "the legacy era selects its flux baseline by physical window "
            "(flux_baseline_window) or by the historical index ranges."
        )


def vest_magnetics_time_window(shot: int, config: VestMagneticsProcessingConfig | None = None) -> np.ndarray:
    """The output time window a VEST shot's magnetics are resampled onto.

    Parameters
    ----------
    shot : int
        Shot number, which selects the acquisition era [-].
    config : VestMagneticsProcessingConfig, optional
        Processing settings; the shipped defaults when omitted [-].

    Returns
    -------
    np.ndarray
        The sample times of the output window [s].

    Convention
    ----------
    A slice of the shot's own timebase, chosen by era rather than by inspecting
    the record. The window and baseline indices are legacy acquisition-era policy, not a
        signal-quality decision: which samples apply depends on the shot number,
        and the eras are recorded in ``vest.yaml`` rather than inferred from the
        data.

    Defaults
    --------
    Every index is an acquisition-era value. Two eras exist: a transient block of
    six shots and everything from a later shot onward share one window, and
    earlier shots take another. A configured override from ``vest.yaml``
    supersedes both; a directly constructed record keeps the historical thresholds
    so old behaviour is reproducible.

    Applicability
    -------------
    VEST-specific.  The windows are VEST acquisition policy, tied to that
    machine's digitizer timebase and its shot-numbering eras.

    Limitations
    -----------
    Selects by shot number alone. A shot whose acquisition differed from its era's
    policy gets that era's window regardless.

    Provenance
    ----------
    .. [1] ``vest.yaml`` ``magnetics.equilibrium_magnetics.processing.window``
       (issue #195), which carries the era table and supersedes the hard-coded
       thresholds.
    """
    cfg = config or DEFAULT_VEST_MAGNETICS_PROCESSING
    index_start, index_end, _ = cfg.window_for_shot(int(shot))
    return cfg.timebase()[index_start : index_end + 1]


def vest_b_field_pol_probe_legacy(
    time: np.ndarray,
    raw: np.ndarray,
    calibration: float,
    *,
    shot: int,
    config: VestMagneticsProcessingConfig | None = None,
    allow_zero_fallback: bool | None = None,
    channel: str | int | None = None,
) -> np.ndarray:
    """Poloidal field at a VEST probe, from its raw record.

    Parameters
    ----------
    time : array_like
        Time base [s].
    raw : array_like
        Raw digitizer record [V].
    calibration : float
        Calibration factor for the probe [-].
    shot : int
        Shot number, which selects the baseline era [-].
    config : VestMagneticsProcessingConfig, optional
        Processing settings [-].
    allow_zero_fallback : bool, optional
        Whether to fall back to zero baseline with a warning when the baseline window
        is degenerate (fewer than 2 valid samples). Defaults to config.allow_zero_fallback [-].
    channel : str or int, optional
        Channel identifier for diagnostic error reporting [-].

    Returns
    -------
    np.ndarray
        Poloidal field at the probe [T].

    Processing steps
    ----------------
    1. Low-pass the raw record on its native fast grid.
    2. Calibrate it.
    3. Integrate in time, carrying Faraday's minus sign.
    4. Fit a linear baseline over the era's leading samples and subtract it.

    Convention
    ----------
    The low pass comes **before** the integration and is an anti-alias step: the
    record is conditioned on the fast acquisition grid below the output window's
    Nyquist frequency, so resampling onto that window afterwards does not fold
    high-frequency content into the result.

    The baseline is the era's **leading** samples, a count rather than a physical
    window, so it means a different interval at a different rate.

    The window and baseline indices are legacy acquisition-era policy, not a
        signal-quality decision: which samples apply depends on the shot number,
        and the eras are recorded in ``vest.yaml`` rather than inferred from the
        data.

    Defaults
    --------
    The cut-off, the tap count and the baseline length are acquisition-era values
    tied to the VEST fast grid and its output window.

    Applicability
    -------------
    VEST-specific.  Written for the VEST poloidal probe chain, its digitizer rates
    and its shot eras.

    Limitations
    -----------
    Assumes the leading samples precede the discharge. Raises `DegenerateBaselineWindowError`
    if fewer than two valid baseline samples are available (issue #639), unless
    `allow_zero_fallback=True` is opted into.

    Provenance
    ----------
    .. [1] The legacy VEST magnetics workflow, whose stage order and baseline
       indices are preserved; ``vest.yaml`` ``magnetics`` for the era table.
    """
    cfg = config or DEFAULT_VEST_MAGNETICS_PROCESSING
    fallback = cfg.allow_zero_fallback if allow_zero_fallback is None else bool(allow_zero_fallback)
    _, _, baseline_end = cfg.window_for_shot(int(shot))
    time = np.asarray(time, dtype=float)
    raw = np.asarray(raw, dtype=float)
    if time.size <= 1 or raw.size <= 1:
        raise ValueError("VEST poloidal-field processing requires at least two samples")

    lowpass = signal.firwin(
        cfg.lowpass_taps,
        cfg.lowpass_cutoff,
        pass_zero="lowpass",
        fs=cfg.fast_sample_rate,
    )
    filtered = signal.lfilter(lowpass, 1, raw)
    calibrated = _calibrated_signal(filtered, float(calibration), cfg.calibration_mode)
    integrated = -cumtrapz_compat(calibrated, x=time, initial=0)
    baseline_indices = np.arange(min(baseline_end, integrated.size))
    baseline = _linear_baseline(
        time,
        integrated,
        baseline_indices,
        allow_zero_fallback=fallback,
        shot=int(shot),
        channel="b_field_pol_probe" if channel is None else channel,
        window_desc=f"leading 0:{baseline_end} samples",
    )
    return integrated - baseline


def vest_flux_loop_voltage(
    raw: np.ndarray,
    calibration: float,
    *,
    config: VestMagneticsProcessingConfig | None = None,
) -> np.ndarray:
    """Calibrated flux-loop terminal voltage, before integration.

    Parameters
    ----------
    raw : array_like
        Raw digitizer record [V].
    calibration : float
        Calibration factor for the loop [-].
    config : VestMagneticsProcessingConfig, optional
        Processing settings; the shipped defaults when omitted [-].

    Returns
    -------
    np.ndarray
        The calibrated terminal voltage [V].

    Convention
    ----------
    Calibration divides by default rather than multiplying, which is the donor's
    convention and the opposite of what a gain usually means; the mode is
    configurable and any other value is refused.

    This is the single definition of a calibrated flux-loop voltage, shared by
    :func:`vest_flux_loop_legacy` and the voltage the ODS mapper stores, so the
    two cannot drift apart.

    Applicability
    -------------
    VEST-specific.  The calibration convention is that of the VEST flux-loop
    digitizer chain.

    Limitations
    -----------
    Voltage only. Nothing is integrated and no baseline is removed here.

    Provenance
    ----------
    .. [1] Issue #209, which established this as the single definition shared with
       the ODS mapping.
    """
    cfg = config or DEFAULT_VEST_MAGNETICS_PROCESSING
    return _calibrated_signal(raw, float(calibration), cfg.calibration_mode)


def vest_flux_loop_flux_from_voltage(
    time: np.ndarray,
    voltage: np.ndarray,
    *,
    flux_loop_number: int,
    config: VestMagneticsProcessingConfig | None = None,
    allow_zero_fallback: bool | None = None,
    shot: int | None = None,
    channel: str | int | None = None,
) -> np.ndarray:
    """Poloidal flux through a VEST flux loop, from its calibrated voltage.

    Parameters
    ----------
    time : array_like
        Time base of the voltage record [s].
    voltage : array_like
        Calibrated terminal voltage [V].
    flux_loop_number : int
        Which loop, since some take their own baseline window [-].
    config : VestMagneticsProcessingConfig, optional
        Processing settings; the shipped defaults when omitted [-].
    allow_zero_fallback : bool, optional
        Whether to fall back to zero baseline with a warning when the baseline window
        is degenerate (fewer than 2 valid samples). Defaults to config.allow_zero_fallback [-].
    shot : int, optional
        Shot number for diagnostic error reporting [-].
    channel : str or int, optional
        Channel identifier for diagnostic error reporting [-].

    Returns
    -------
    np.ndarray
        Flux through the loop, per radian [Wb/rad].

    Processing steps
    ----------------
    1. Integrate the voltage in time, carrying Faraday's minus sign.
    2. Divide by two pi to get flux per radian.
    3. Fit a linear baseline over the era's indices and subtract it.

    Convention
    ----------
    **Per radian**, which is what the ODS mapper expects and multiplies back by
    two pi. The minus sign is Faraday's and is applied before the baseline, so the
    constant of integration is absorbed into the fitted line rather than chosen.

    The window and baseline indices are legacy acquisition-era policy, not a
        signal-quality decision: which samples apply depends on the shot number,
        and the eras are recorded in ``vest.yaml`` rather than inferred from the
        data.

    Defaults
    --------
    The baseline indices are acquisition-era values in a four-way ladder: an
    explicit sample count for the native-DAQ era, then a physical time window,
    then a per-loop exception for three specific loops, then the two-segment
    default. The native-DAQ and later slow-DAQ windows are legacy compatibility
    values reproducing the donor's own indices.

    Applicability
    -------------
    VEST-specific.  Carries VEST's acquisition eras, its per-loop baseline
    exceptions, and its digitizer timebase.

    Limitations
    -----------
    Raises `DegenerateBaselineWindowError` when fewer than two valid baseline samples
    are available in the configured window (issue #639), unless `allow_zero_fallback=True`
    is explicitly specified. The per-loop exception is by loop number, so a rewiring that
    changes which loop is which invalidates it.

    Provenance
    ----------
    .. [1] ``VEST_MagneticSignalProcessing2.m`` for the native-DAQ era, whose
       leading-sample baseline indices are reproduced; and the slow-DAQ era's own
       window, expressed in physical seconds here so it stays correct at any
       sample rate.
    .. [2] ``vest.yaml`` ``magnetics`` for the era table (issue #195).
    """
    cfg = config or DEFAULT_VEST_MAGNETICS_PROCESSING
    fallback = cfg.allow_zero_fallback if allow_zero_fallback is None else bool(allow_zero_fallback)
    time = np.asarray(time, dtype=float)
    calibrated = np.asarray(voltage, dtype=float)
    if time.size <= 1 or calibrated.size <= 1:
        raise ValueError("VEST flux-loop processing requires at least two samples")
    integrated = -cumtrapz_compat(calibrated, x=time, initial=0)
    if cfg.flux_output_per_radian:
        integrated = integrated / (2 * np.pi)

    window_desc: str | None = None
    if cfg.flux_baseline_samples is not None:
        # Native-DAQ era (VEST_MagneticSignalProcessing2.m): flux loops moved
        # onto the 250 kHz acquisition and take the same leading-sample
        # baseline as the B-pol probes -- MATLAB
        # `polyfit(timeFastFL(index_Bz_start:index_Bz_end), ...)` with
        # index_Bz_start = 1, index_Bz_end = 1750.
        baseline_indices = np.arange(min(int(cfg.flux_baseline_samples), integrated.size))
        window_desc = f"samples 0:{cfg.flux_baseline_samples}"
    elif cfg.flux_baseline_window is not None:
        # Shot >= 43685 on the slow-DAQ era: MATLAB
        # `index_FL_start = 6001 (0.24 s), index_FL_end = 6500 (0.26 s)`.
        # Expressed in physical seconds so it stays correct regardless of the
        # loop's native sample rate.
        window_start, window_end = (float(bound) for bound in cfg.flux_baseline_window)
        baseline_indices = np.flatnonzero((time >= window_start) & (time <= window_end))
        window_desc = f"time window [{window_start:.6g}, {window_end:.6g}] s"
    elif int(flux_loop_number) in cfg.flux_baseline_late_loop_numbers:
        baseline_indices = np.arange(cfg.flux_baseline_late_start, min(cfg.flux_baseline_late_end, integrated.size))
        window_desc = f"late loop samples {cfg.flux_baseline_late_start}:{cfg.flux_baseline_late_end}"
    else:
        first = np.arange(cfg.flux_baseline_first_start, min(cfg.flux_baseline_first_end, integrated.size))
        second = np.arange(cfg.flux_baseline_second_start, min(cfg.flux_baseline_second_end, integrated.size))
        baseline_indices = np.concatenate((first, second))
        window_desc = (
            f"default loop samples {cfg.flux_baseline_first_start}:{cfg.flux_baseline_first_end} "
            f"and {cfg.flux_baseline_second_start}:{cfg.flux_baseline_second_end}"
        )

    channel_name = channel if channel is not None else f"flux_loop_{flux_loop_number}"
    baseline = _linear_baseline(
        time,
        integrated,
        baseline_indices,
        allow_zero_fallback=fallback,
        shot=shot,
        channel=channel_name,
        window_desc=window_desc,
    )
    return integrated - baseline


def vest_flux_loop_legacy(
    time: np.ndarray,
    raw: np.ndarray,
    calibration: float,
    *,
    flux_loop_number: int,
    config: VestMagneticsProcessingConfig | None = None,
    allow_zero_fallback: bool | None = None,
    shot: int | None = None,
    channel: str | int | None = None,
) -> np.ndarray:
    """Poloidal flux through a VEST flux loop, from its raw record.

    Parameters
    ----------
    time : array_like
        Time base [s].
    raw : array_like
        Raw digitizer record [V].
    calibration : float
        Calibration factor for the loop [-].
    flux_loop_number : int
        Which loop [-].
    config : VestMagneticsProcessingConfig, optional
        Processing settings [-].
    allow_zero_fallback : bool, optional
        Whether to fall back to zero baseline with a warning when the baseline window
        is degenerate (fewer than 2 valid samples). Defaults to config.allow_zero_fallback [-].
    shot : int, optional
        Shot number for diagnostic error reporting [-].
    channel : str or int, optional
        Channel identifier for diagnostic error reporting [-].

    Returns
    -------
    np.ndarray
        Flux through the loop, per radian [Wb/rad].

    Processing steps
    ----------------
    1. Calibrate the raw record into a terminal voltage.
    2. Integrate, normalize per radian and subtract the era's baseline.

    Convention
    ----------
    The composition of :func:`vest_flux_loop_voltage` and
    :func:`vest_flux_loop_flux_from_voltage`, kept as a single call because the
    donor workflow had one. Same per-radian output and same era policy as those.

    Applicability
    -------------
    VEST-specific.  Inherits the calibration convention and the acquisition eras
    of the two functions it composes.

    Limitations
    -----------
    Inherits both. Raises `DegenerateBaselineWindowError` on degenerate baseline
    windows (issue #639) unless `allow_zero_fallback=True`.

    Provenance
    ----------
    .. [1] The legacy VEST magnetics workflow; see the two functions this
       composes for the donor indices.
    """
    time = np.asarray(time, dtype=float)
    raw = np.asarray(raw, dtype=float)
    if time.size <= 1 or raw.size <= 1:
        raise ValueError("VEST flux-loop processing requires at least two samples")

    voltage = vest_flux_loop_voltage(raw, calibration, config=config)
    return vest_flux_loop_flux_from_voltage(
        time,
        voltage,
        flux_loop_number=flux_loop_number,
        config=config,
        allow_zero_fallback=allow_zero_fallback,
        shot=shot,
        channel=channel,
    )


@dataclass(frozen=True)
class VestEquilibriumMagneticsResult:
    """Processed VEST equilibrium magnetics waveforms with native voltages.

    `flux_loops`/`probes` are resampled onto `time` (the MD output window),
    while `flux_loop_voltage`/`flux_loop_voltage_time` keep the native
    acquisition timebase of each flux-loop channel. All flux-loop lists are
    index-aligned: an unavailable channel contributes an empty array to each
    of them rather than shifting later channels.
    """

    time: np.ndarray
    flux_loops: list[np.ndarray]
    probes: list[np.ndarray]
    flux_loop_voltage_time: list[np.ndarray]
    flux_loop_voltage: list[np.ndarray]


def vest_equilibrium_magnetics_detailed(
    shot: int,
    channels: Sequence[dict],
    loader: Callable[[int, int], tuple[np.ndarray, np.ndarray] | None],
    *,
    indices: Sequence[int] | None = None,
    config: VestMagneticsProcessingConfig | None = None,
    allow_missing: bool = False,
    allow_zero_fallback: bool | None = None,
) -> VestEquilibriumMagneticsResult:
    """Process every magnetic channel of one VEST shot onto the output window.

    Parameters
    ----------
    shot : int
        Shot number, which selects the acquisition era [-].
    channels : sequence
        The channels to process, each naming its kind and calibration [-].
    loader : callable
        Returns the raw record and its time base for a channel [-].
    indices : sequence of int, optional
        Which channels to process; all of them when omitted [-].
    config : VestMagneticsProcessingConfig, optional
        Processing settings [-].
    allow_missing : bool, optional
        Whether a channel the loader cannot supply is tolerated [-].
    allow_zero_fallback : bool, optional
        Whether to fall back to zero baseline with a warning when a channel encounters
        a degenerate baseline window. Defaults to config.allow_zero_fallback [-].

    Returns
    -------
    VestEquilibriumMagneticsResult
        The processed flux in weber per radian and field in tesla on the output
        window, the calibrated voltages at their native rate, and the window
        itself [-].

    Processing steps
    ----------------
    1. Resolve the shot's output window from its era.
    2. For each channel, load the raw record and dispatch on its kind, probe or
       flux loop, to the matching processing chain.
    3. Resample the result onto the output window.
    4. Keep the calibrated voltages at their native rate alongside.

    Convention
    ----------
    **Index alignment is an invariant.** An unavailable channel contributes an
    empty array rather than being dropped, so channel positions never shift and a
    caller can index the result by the same number it indexed the input by. That
    is why a missing channel is not simply skipped.

    Voltages are kept at the native acquisition rate while fluxes and fields are
    resampled, because the resampling is what the equilibrium chain needs and the
    voltage is what a diagnostic comparison needs.

    Applicability
    -------------
    VEST-specific.  Orchestrates the VEST legacy chain, its channel kinds, its
    calibrations and its acquisition eras.

    Limitations
    -----------
    Inherits every limitation of the per-channel chains. Raises `DegenerateBaselineWindowError`
    if any channel has a degenerate baseline window (issue #639), unless
    `allow_zero_fallback=True` is opted into. With missing channels tolerated, an empty entry
    is indistinguishable from a genuinely zero record without checking its length.

    Provenance
    ----------
    .. [1] The legacy ``vfit_equilibrium_magnetics`` workflow this reproduces;
       ``vest.yaml`` ``magnetics`` for the era policy.
    """
    from vaft.database import raw as raw_db

    cfg = config or DEFAULT_VEST_MAGNETICS_PROCESSING
    _validate_daq_mode(cfg)
    fallback = cfg.allow_zero_fallback if allow_zero_fallback is None else bool(allow_zero_fallback)
    channel_rows = list(channels)
    if indices is not None:
        channel_rows = [channel_rows[int(index)] for index in indices]

    magnetics_time = vest_magnetics_time_window(shot, cfg)
    data_flux_loops: list[np.ndarray] = []
    data_probes: list[np.ndarray] = []
    flux_loop_voltage_time: list[np.ndarray] = []
    flux_loop_voltage: list[np.ndarray] = []
    flux_loop_counter = 0

    for channel in channel_rows:
        field_code = int(channel["field_code"])
        calibration = float(channel["calibration"])
        kind = str(channel["kind"])
        loaded = loader(int(shot), field_code)

        if kind == "flux_loop":
            flux_loop_counter += 1

        try:
            source_time, source_data = raw_db.require_signal(
                loaded,
                shot=shot,
                field=field_code,
                signal_name=str(channel.get("name", kind)),
            )
        except raw_db.RawSignalUnavailableError:
            if not allow_missing:
                raise
            missing = np.array([], dtype=float)
            if kind == "b_field_pol_probe":
                data_probes.append(missing)
            else:
                data_flux_loops.append(missing)
                flux_loop_voltage_time.append(missing)
                flux_loop_voltage.append(missing)
            continue

        if kind == "b_field_pol_probe":
            channel_name = str(channel.get("name", f"b_field_pol_probe_{field_code}"))
            processed_full = vest_b_field_pol_probe_legacy(
                source_time,
                source_data,
                calibration,
                shot=shot,
                config=cfg,
                allow_zero_fallback=fallback,
                channel=channel_name,
            )
            # anti-alias: vest_b_field_pol_probe_legacy low-passes at 2.5 kHz on
            # the 250 kHz source grid, below the 25 kHz target's Nyquist.
            data_probes.append(np.interp(magnetics_time, source_time, processed_full))
            continue

        source_time = np.asarray(source_time, dtype=float)
        source_data = np.asarray(source_data, dtype=float)
        voltage = vest_flux_loop_voltage(source_data, calibration, config=cfg)
        channel_name = str(channel.get("name", f"flux_loop_{flux_loop_counter}"))
        processed_full = vest_flux_loop_flux_from_voltage(
            source_time,
            voltage,
            flux_loop_number=flux_loop_counter,
            config=cfg,
            allow_zero_fallback=fallback,
            shot=shot,
            channel=channel_name,
        )
        # anti-alias: the flux is the integral of an already low-passed loop
        # voltage, so it is band-limited on the source grid.
        data_flux_loops.append(np.interp(magnetics_time, source_time, processed_full))
        flux_loop_voltage_time.append(source_time)
        flux_loop_voltage.append(voltage)

    return VestEquilibriumMagneticsResult(
        time=magnetics_time,
        flux_loops=data_flux_loops,
        probes=data_probes,
        flux_loop_voltage_time=flux_loop_voltage_time,
        flux_loop_voltage=flux_loop_voltage,
    )


def vest_equilibrium_magnetics_signals(
    shot: int,
    channels: Sequence[dict],
    loader: Callable[[int, int], tuple[np.ndarray, np.ndarray] | None],
    *,
    indices: Sequence[int] | None = None,
    config: VestMagneticsProcessingConfig | None = None,
    allow_missing: bool = False,
    allow_zero_fallback: bool | None = None,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """The processed magnetics of one VEST shot, as the three-array legacy view.

    Parameters
    ----------
    shot : int
        Shot number [-].
    channels : sequence
        The channels to process [-].
    loader : callable
        Returns the raw record and its time base for a channel [-].
    indices : sequence of int, optional
        Which channels to process [-].
    config : VestMagneticsProcessingConfig, optional
        Processing settings [-].
    allow_missing : bool, optional
        Whether an unavailable channel is tolerated [-].
    allow_zero_fallback : bool, optional
        Whether to fall back to zero baseline with a warning when a channel encounters
        a degenerate baseline window. Defaults to config.allow_zero_fallback [-].

    Returns
    -------
    tuple of np.ndarray
        The output window in seconds, the fluxes in weber per radian and the
        fields in tesla [-].

    Convention
    ----------
    A backward-compatible view of :func:`vest_equilibrium_magnetics_detailed`,
    returning the three arrays the older callers expect and discarding the native-
    rate voltages. New code should call the detailed form, which keeps them.

    Applicability
    -------------
    VEST-specific.  The same chain and eras as the function it wraps.

    Limitations
    -----------
    Inherits everything from that function, and additionally discards the
    native-rate voltages, which cannot then be recovered.

    Provenance
    ----------
    .. [1] :func:`vest_equilibrium_magnetics_detailed`, which does the work.
    """
    result = vest_equilibrium_magnetics_detailed(
        shot,
        channels,
        loader,
        indices=indices,
        config=config,
        allow_missing=allow_missing,
        allow_zero_fallback=allow_zero_fallback,
    )
    return result.time, result.flux_loops, result.probes


# Pre-rename name, kept as a plain alias: `vaft.process` re-exports this
# module with `from .magnetics import *`, so `vaft.process.vest_md_signals`
# must keep working for existing callers.
vest_md_signals = vest_equilibrium_magnetics_signals


def _firwin_order(sample_rate: float) -> int:
    order = int(float(sample_rate) * 1e-3)
    return order + 1 if order % 2 == 0 else order


def mirnov_preprocess_signal(
    data: np.ndarray,
    *,
    sample_rate: float = 250_000.0,
    high_pass_cutoff: float | None = 2_000.0,
    low_pass_cutoff: float | None = 90_000.0,
    amplifier_gain: float = 1.0,
    filter_order: int | None = None,
) -> np.ndarray:
    """Condition a Mirnov coil signal for mode analysis.

    Parameters
    ----------
    data : array_like
        Raw coil record [V].
    sample_rate : float, optional
        Sample rate [Hz].
    high_pass_cutoff : float, optional
        Lower edge of the passband [Hz].
    low_pass_cutoff : float, optional
        Upper edge [Hz].
    amplifier_gain : float, optional
        Gain divided out of the record [-].
    filter_order : int, optional
        Filter order; derived from the sample rate when omitted [-].

    Returns
    -------
    np.ndarray
        The conditioned record [V].

    Processing steps
    ----------------
    1. Remove the record's mean.
    2. Band-pass it with a windowed filter.
    3. Divide out the amplifier gain.

    Convention
    ----------
    Band-passed rather than low-passed, because mode analysis wants neither the
    slow equilibrium evolution below the band nor the noise above it. **The
    output is still a time derivative**: nothing here integrates, so a spectral
    index taken from this is a derivative index, not a field one.

    Defaults
    --------
    The sample rate and the two band edges are acquisition-era values, the VEST
    Mirnov amplifier chain's own. The order is derived from the sample rate, one
    tap per millisecond, and forced odd, which is a numerical convenience keeping
    the filter symmetric.

    Applicability
    -------------
    Machine-independent.  The defaults are VEST's chain; every one is an argument.

    Limitations
    -----------
    The derived order scales with the sample rate, so a very high rate gives a
    long filter and a correspondingly long transient at each end.

    Provenance
    ----------
    .. [1] The legacy ``vest_osc`` Mirnov preprocessing chain, whose stage order
       and band edges this reproduces.
    """
    values = np.asarray(data, dtype=float)
    if values.size == 0:
        return values.copy()
    if amplifier_gain == 0:
        raise ValueError("amplifier_gain must be non-zero.")

    processed = values - float(np.nanmean(values))
    order = int(filter_order) if filter_order is not None else _firwin_order(sample_rate)
    nyquist = 0.5 * float(sample_rate)

    for cutoff, pass_zero in ((high_pass_cutoff, False), (low_pass_cutoff, True)):
        if cutoff is None:
            continue
        cutoff = float(cutoff)
        if cutoff <= 0 or cutoff >= nyquist:
            continue
        taps = signal.firwin(order, cutoff, fs=sample_rate, pass_zero=pass_zero, window="hann")
        if processed.size > 3 * taps.size:
            processed = signal.filtfilt(taps, 1.0, processed)
        else:
            processed = signal.lfilter(taps, 1.0, processed)

    return processed / float(amplifier_gain)


def _sample_rate_from_time(time: np.ndarray, default: float = 250_000.0) -> float:
    if time.size < 2:
        return float(default)
    dt = float(np.nanmedian(np.diff(time)))
    if not np.isfinite(dt) or dt <= 0:
        return float(default)
    return 1.0 / dt


def mirnov_spectrogram(
    time: np.ndarray,
    data: np.ndarray,
    *,
    sample_rate: float | None = None,
    window_size: int = 500,
    time_resolution: int = 1,
    time_range: tuple[float, float] | None = None,
) -> MirnovSpectrogramResult:
    """Sliding-window amplitude spectrogram of a Mirnov signal.

    Parameters
    ----------
    time : array_like
        Time base [s].
    data : array_like
        Conditioned coil record [V].
    sample_rate : float, optional
        Sample rate; derived from the time base when omitted [Hz].
    window_size : int, optional
        Window length in samples [-].
    time_resolution : int, optional
        Step between windows, in samples [-].
    time_range : sequence of float, optional
        Analysis window as a start and end time [s].

    Returns
    -------
    MirnovSpectrogramResult
        The time and frequency axes and the amplitude map [-].

    Convention
    ----------
    **Amplitude, not power**: the transform is normalized so a sinusoid appears at
    its own amplitude, which is the single-sided convention that makes a mode's
    size directly readable. That differs from the power spectral density in the
    fluctuation module, which is squared and per hertz.

    The field names match that module's own spectrogram result, so the plotting
    layer accepts either.

    Defaults
    --------
    The window and step are validated-workflow defaults from the legacy Mirnov
    routine.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Fixed window, so the time and frequency resolution trade off once for the
    whole map. Expects a conditioned input; run
    :func:`mirnov_preprocess_signal` first.

    Provenance
    ----------
    .. [1] The legacy ``vest_mirnov`` routine, whose window and normalization this
       reproduces.
    """
    time = np.asarray(time, dtype=float)
    values = np.asarray(data, dtype=float)
    if time.size != values.size:
        raise ValueError("time and data must have the same length.")
    if time.size == 0:
        return MirnovSpectrogramResult(np.array([]), np.array([]), np.empty((0, 0)))

    window_size = int(window_size)
    if window_size <= 1 or window_size % 2:
        raise ValueError("window_size must be an even integer greater than 1.")
    step = max(1, int(time_resolution))
    half_window = window_size // 2

    if time_range is None:
        first = half_window - 1
        last = values.size - half_window - 1
    else:
        first = int(np.searchsorted(time, float(time_range[0]), side="left"))
        last = int(np.searchsorted(time, float(time_range[1]), side="left"))
    centers = np.arange(first, last + 1, step, dtype=int)
    centers = centers[(centers - half_window + 1 >= 0) & (centers + half_window + 1 <= values.size)]
    if centers.size == 0:
        frequencies = np.fft.rfftfreq(window_size, d=1.0 / (sample_rate or _sample_rate_from_time(time)))
        return MirnovSpectrogramResult(np.array([]), frequencies, np.empty((frequencies.size, 0)))

    windows = np.empty((window_size, centers.size), dtype=float)
    for column, center in enumerate(centers):
        windows[:, column] = values[center - half_window + 1 : center + half_window + 1]

    fs = float(sample_rate) if sample_rate is not None else _sample_rate_from_time(time)
    tapered = windows * signal.windows.hann(window_size)[:, np.newaxis]
    spectrum = np.fft.rfft(tapered, axis=0)
    magnitude = 2.0 * np.abs(spectrum / window_size)
    frequencies = np.fft.rfftfreq(window_size, d=1.0 / fs)
    return MirnovSpectrogramResult(time[centers], frequencies, magnitude)


def toroidal_mode_analysis(
    signal_a: np.ndarray,
    signal_b: np.ndarray,
    *,
    sample_rate: float = 250_000.0,
    phase_geometry: float = np.pi / 6,
    peak_threshold: float = 0.1,
    sensor_count: int = 4,
    nperseg: int | None = None,
) -> ToroidalModeResult:
    """Toroidal mode number from the cross-spectral phase of two coils.

    Parameters
    ----------
    signal_a : array_like
        Conditioned record from the first coil [V].
    signal_b : array_like
        Conditioned record from the second [V].
    sample_rate : float, optional
        Sample rate [Hz].
    phase_geometry : float, optional
        Toroidal angle of the second coil minus the first coil, phi_b - phi_a [rad].
    peak_threshold : float, optional
        Fraction of the maximum a peak must reach to be reported [-].
    sensor_count : int, optional
        Number of sensors averaged, which sets the coherence significance [-].
    nperseg : int, optional
        Segment length of the cross-spectral estimate [-].

    Returns
    -------
    ToroidalModeResult
        The coherent peaks with their frequency, phase and mode number [-].

    Processing steps
    ----------------
    1. Estimate the cross-spectral density and the coherence of the pair.
    2. Keep frequencies whose coherence exceeds the significance level.
    3. Find the peaks among them above the threshold.
    4. Divide the negative of each peak's phase by the toroidal separation to get its mode number.

    Convention
    ----------
    **The mode number carries a minus sign**: ``n_raw = -phase / phase_geometry``.
    Under standard right-handed cylindrical coordinates (R, phi, Z) with phi
    counter-clockwise from above, a mode propagating in the co-current (+phi)
    direction has phase decreasing with increasing phi (theta(phi) = theta_0 -
    n * phi, so d(theta)/d(phi) = -n). A sensor at Delta_phi > 0 lags the
    reference sensor by Delta_theta = -n * Delta_phi, so n = -Delta_theta /
    Delta_phi. This agrees with the model fitted by
    :func:`toroidal_phase_fit_at_time`.

    The coherence threshold is the 95 percent significance level for a
    magnitude-squared coherence averaged over the given number of sensors, so it
    tightens as more are averaged rather than being a fixed number.

    Defaults
    --------
    The sample rate, the toroidal separation and the sensor count are
    machine-specific: they are the VEST array's spacing and size. The peak
    threshold is hard-coded with no recorded derivation.

    Applicability
    -------------
    Machine-independent.  The array geometry is an argument.

    Limitations
    -----------
    Two coils cannot resolve aliasing: a mode and one differing by a full turn
    over the separation give the same phase. Nothing here reports the degeneracy,
    unlike the soft X-ray equivalent, which returns every candidate.

    Provenance
    ----------
    .. [1] The magnitude-squared coherence significance level for an averaged
       estimate, which sets the threshold.
    """
    a = np.asarray(signal_a, dtype=float)
    b = np.asarray(signal_b, dtype=float)
    if a.size != b.size:
        raise ValueError("signal_a and signal_b must have the same length.")
    if a.size < 2:
        empty = np.array([])
        return ToroidalModeResult(empty, empty, empty, empty, empty, empty, empty.astype(int), empty, empty, empty)
    if np.isclose(float(phase_geometry), 0.0):
        raise ValueError("phase_geometry must be non-zero.")

    segment = min(a.size, int(nperseg) if nperseg is not None else 256)
    frequencies, cross_power = csd(a, b, fs=sample_rate, nperseg=segment)
    _, coherence_values = coherence(a, b, fs=sample_rate, nperseg=segment)
    phase = np.angle(cross_power)
    n_raw = -phase / float(phase_geometry)
    n_rounded = np.round(n_raw)

    power_abs = np.abs(cross_power)
    if power_abs.size == 0 or float(np.max(power_abs)) == 0.0:
        peak_indices = np.array([], dtype=int)
    else:
        peaks, _ = find_peaks(power_abs, height=float(peak_threshold) * float(np.max(power_abs)))
        coherence_threshold = np.tanh(1.96 / np.sqrt(max(1.0, 2.0 * float(sensor_count) - 2.0)))
        peak_indices = peaks[coherence_values[peaks] > coherence_threshold]

    total_power = float(np.sum(power_abs))
    relative_power = power_abs[peak_indices] / total_power if total_power > 0 else np.zeros(peak_indices.size)
    return ToroidalModeResult(
        frequency=frequencies[peak_indices],
        n=n_rounded[peak_indices],
        power=relative_power,
        phase=phase[peak_indices],
        spectrum_frequency=frequencies,
        cross_power=cross_power,
        peak_indices=peak_indices,
        n_raw=n_raw,
        n_rounded=n_rounded,
        coherence=coherence_values,
    )


def _wrap_phase_radians(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=float) + np.pi) % (2 * np.pi) - np.pi


def _fit_wrapped_toroidal_n(
    toroidal_angle: np.ndarray,
    phase: np.ndarray,
    candidate_n: np.ndarray,
) -> tuple[int, float, float, np.ndarray]:
    best_n = 0
    best_intercept = 0.0
    best_error = np.inf
    best_fit = np.zeros_like(phase)
    for n_value in candidate_n:
        residual_offset = phase + float(n_value) * toroidal_angle
        intercept = float(np.angle(np.mean(np.exp(1j * residual_offset))))
        fitted = _wrap_phase_radians(intercept - float(n_value) * toroidal_angle)
        residual = _wrap_phase_radians(phase - fitted)
        rms_error = rms(residual)
        if rms_error < best_error:
            best_n = int(n_value)
            best_intercept = intercept
            best_error = rms_error
            best_fit = fitted
    return best_n, best_intercept, best_error, best_fit


def toroidal_phase_fit_at_time(
    time: np.ndarray,
    signals: np.ndarray,
    toroidal_angle: np.ndarray,
    *,
    center_time: float,
    sample_rate: float | None = None,
    window_size: int = 500,
    frequencies: Sequence[float] | None = None,
    num_modes: int = 2,
    candidate_n: Sequence[int] = tuple(range(-6, 7)),
    peak_threshold: float = 0.1,
) -> ToroidalPhaseFitResult:
    """Toroidal mode numbers from a wrapped-phase fit across an array, at one time.

    Parameters
    ----------
    time : array_like
        Time base [s].
    signals : array_like
        Conditioned records, one row per coil [V].
    toroidal_angle : array_like
        Toroidal angle of each coil [rad].
    center_time : float
        Centre of the analysis window [s].
    sample_rate : float, optional
        Sample rate; derived from the time base when omitted [Hz].
    window_size : int, optional
        Window length in samples [-].
    frequencies : array_like, optional
        Frequencies to fit; the strongest peaks when omitted [Hz].
    num_modes : int, optional
        How many mode lines to fit [-].
    candidate_n : sequence of int, optional
        Mode numbers to consider [-].
    peak_threshold : float, optional
        Fraction of the maximum a peak must reach [-].

    Returns
    -------
    ToroidalPhaseFitResult
        The fitted mode number, intercept and residual for each mode line [-].

    Processing steps
    ----------------
    1. Take the window centred on the requested time.
    2. Choose the frequencies to fit, from the caller or from the strongest peaks.
    3. For each, measure the phase at every coil.
    4. Score each candidate mode number by its wrapped residual against the
       measured phases and keep the best.

    Convention
    ----------
    **The model carries a minus sign**: the fitted phase decreases with increasing
    toroidal angle for a positive mode number (``fitted = intercept - n * toroidal_angle``).
    Under standard right-handed cylindrical coordinates (R, phi, Z), this
    corresponds to a perturbation propagating in the positive phi (co-current)
    direction, in agreement with :func:`toroidal_mode_analysis`.

    Phases are wrapped to a single turn and the intercept is a circular mean, so
    the fit is insensitive to where the branch cut falls, which a plain average
    would not be.

    Defaults
    --------
    The window length and the candidate range are validated-workflow defaults from
    the legacy array analysis. The peak threshold is hard-coded with no recorded
    derivation.

    Applicability
    -------------
    Machine-independent.  The array's angles are an argument.

    Limitations
    -----------
    Fits at one time only, so a mode whose number changes through the shot needs
    repeated calls. Resolving a mode number needs enough coils spread over enough
    angle; a small or clustered array leaves candidates degenerate, and the fit
    returns the best-scoring one without reporting that.

    Provenance
    ----------
    .. [1] The wrapped-phase array fit of the legacy VEST Mirnov analysis.
    """
    time = np.asarray(time, dtype=float)
    data = np.asarray(signals, dtype=float)
    angles = np.asarray(toroidal_angle, dtype=float)
    candidates = np.asarray(tuple(candidate_n), dtype=int)

    if data.ndim != 2:
        raise ValueError("signals must have shape (n_channels, n_time).")
    if data.shape[0] != angles.size:
        raise ValueError("toroidal_angle length must match the number of signal channels.")
    if data.shape[1] != time.size:
        raise ValueError("signals time dimension must match time length.")
    if data.shape[0] < 2:
        raise ValueError("At least two toroidal channels are required.")
    if candidates.size == 0:
        raise ValueError("candidate_n must contain at least one integer.")

    window_size = int(window_size)
    if window_size <= 1 or window_size % 2:
        raise ValueError("window_size must be an even integer greater than 1.")
    if time.size < window_size:
        raise ValueError("time array is shorter than window_size.")

    center_index = int(np.argmin(np.abs(time - float(center_time))))
    half_window = window_size // 2
    start = center_index - half_window
    stop = start + window_size
    if start < 0:
        start = 0
        stop = window_size
    if stop > time.size:
        stop = time.size
        start = stop - window_size

    fs = float(sample_rate) if sample_rate is not None else _sample_rate_from_time(time)
    window = signal.windows.hann(window_size)
    windowed = data[:, start:stop] * window[np.newaxis, :]
    spectrum = np.fft.rfft(windowed, axis=1)
    spectrum_frequency = np.fft.rfftfreq(window_size, d=1.0 / fs)
    magnitude = np.abs(spectrum)

    if frequencies is None:
        average_magnitude = np.mean(magnitude, axis=0)
        average_magnitude[0] = 0.0
        if np.max(average_magnitude) <= 0:
            selected_indices = np.array([], dtype=int)
        else:
            peaks, props = find_peaks(average_magnitude, height=float(peak_threshold) * float(np.max(average_magnitude)))
            if peaks.size == 0:
                selected_indices = np.array([int(np.argmax(average_magnitude))], dtype=int)
            else:
                order = np.argsort(props["peak_heights"])[::-1]
                selected_indices = peaks[order[: max(1, int(num_modes))]]
    else:
        selected_indices = np.array(
            [int(np.argmin(np.abs(spectrum_frequency - float(freq)))) for freq in frequencies],
            dtype=int,
        )

    modes: list[ToroidalPhaseModeFit] = []
    for frequency_index in selected_indices:
        complex_values = spectrum[:, frequency_index]
        phase = _wrap_phase_radians(np.angle(complex_values))
        amplitude = np.abs(complex_values)
        best_n, intercept, rms_error, fitted = _fit_wrapped_toroidal_n(angles, phase, candidates)
        modes.append(
            ToroidalPhaseModeFit(
                frequency=float(spectrum_frequency[frequency_index]),
                n=best_n,
                intercept=intercept,
                rms_error=rms_error,
                phase=phase,
                fitted_phase=fitted,
                amplitude=amplitude,
            )
        )

    modes.sort(key=lambda item: float(np.mean(item.amplitude)), reverse=True)
    return ToroidalPhaseFitResult(
        time=float(time[center_index]),
        toroidal_angle=angles,
        modes=tuple(modes),
        candidate_n=candidates,
    )

def rogowski_coil_ip(
    time,
    rogowski_raw,
    flux_loop_raw,
    flux_loop_gain=11,
    effective_vessel_res=5.8e-4,
    baseline_onset=0.27,
    baseline_offset=0.28,
    baseline_type='linear',
    baseline_onset_window=500,
    baseline_offset_window=100,
    smooth_window=10
):
    """Plasma current from a Rogowski coil, compensated by a flux-loop reference.

    A Rogowski coil encircles both the plasma and the vessel current, so its
    signal is not the plasma current alone. A flux loop supplies the reference
    that removes the vessel's share.

    Parameters
    ----------
    time : array_like
        Time base [s].
    rogowski_raw : array_like
        Raw Rogowski record, already a current-like signal [A].
    flux_loop_raw : array_like
        Raw flux-loop record used as the vessel reference [V].
    flux_loop_gain : float, optional
        Scales the reference into current units [A/V].
    effective_vessel_res : float, optional
        Effective vessel resistance.  Accepted but **not used** [ohm].
    baseline_onset : float, optional
        Time at which the signals begin to deviate [s].
    baseline_offset : float, optional
        Time by which they have returned [s].
    baseline_type : str, optional
        Which model to fit as the baseline [-].
    baseline_onset_window : int, optional
        Samples before the onset to include in the fit [-].
    baseline_offset_window : int, optional
        Samples after the offset to include [-].
    smooth_window : int, optional
        Smoothing width applied to the reference, odd [-].

    Returns
    -------
    np.ndarray
        The plasma current [A].

    Processing steps
    ----------------
    1. Fit a baseline over the quiet samples either side of the discharge and
       subtract it from both records.
    2. Scale the flux-loop record into a current reference.
    3. Smooth that reference.
    4. Subtract it from the Rogowski record.
    5. Flip the sign if the negative excursion dominates.

    Convention
    ----------
    **Polarity is inferred from the data, not asserted.**  If the result's
    negative peak exceeds its positive one the whole trace is flipped, on the
    assumption that a discharge is a single-signed excursion. This is the one
    place in the module where a sign convention is decided by the signal rather
    than declared, and it is wrong for any record whose largest excursion is not
    the plasma current.

    No integration happens here: the Rogowski input is already current-like.

    Defaults
    --------
    The gain and the two baseline times are machine-specific VEST values. The
    smoothing width and the two window lengths are numerical conveniences. The
    vessel resistance is **hard-coded and unused**: it is accepted for
    compatibility, the dimensional argument for it being a resistance rather than
    an inductance is recorded in issue #214, and no code path reads it.

    Applicability
    -------------
    Machine-independent.  Every machine number is an argument, and the defaults
    are VEST's.

    Limitations
    -----------
    The compensation is a scaled subtraction, not a circuit model, so it removes
    the vessel's share only to the extent that share is proportional to the flux
    loop's signal. The sign inference above can invert a correct trace. Assumes
    the record is quiet either side of the baseline window.

    Provenance
    ----------
    .. [1] Issue #214, which records the dimensional argument for the unused
       vessel resistance and the flux-loop compensation it belongs to.
    """
    # Convert baseline onset/offset in seconds to integer indices
    onset_idx = np.searchsorted(time, baseline_onset)
    offset_idx = np.searchsorted(time, baseline_offset)

    # Define baseline indices
    baseline_indices_rogowski = define_baseline(
        time, onset_idx, baseline_onset_window, offset_idx, baseline_offset_window
    )
    baseline_indices_flux = baseline_indices_rogowski  # same region, typically

    # Subtract baseline from rogowski
    rogowski_corr, rogowski_baseline = subtract_baseline(
        time, rogowski_raw, baseline_indices_rogowski, fitting_opt=baseline_type
    )

    # Subtract baseline from flux loop
    flux_corr, flux_baseline = subtract_baseline(
        time, flux_loop_raw, baseline_indices_flux, fitting_opt=baseline_type
    )

    # Convert flux loop signal to current reference
    # For example: flux_ref = flux_corr * (flux_loop_gain / effective_resistance)
    # -- the divisor is in ohms, not henries; see issue #214.
    # We'll do the simplest version: flux_corr * flux_loop_gain
    flux_ref = flux_corr * flux_loop_gain

    # Smooth the flux loop reference (Savitzky-Golay) if smooth_window > 2
    if smooth_window < 3:
        smooth_window = 3
    if smooth_window % 2 == 0:
        smooth_window += 1

    flux_ref_smooth = savgol_filter(flux_ref, smooth_window, polyorder=1)

    # Final plasma current
    ip = rogowski_corr - flux_ref_smooth

    # If absolute negative peak is larger than the positive peak, invert
    if abs(np.min(ip)) > abs(np.max(ip)):
        ip = -ip

    return time, ip


def b_field_pol_probe_field(
    time,
    raw,
    gain,
    lowpass_param,
    baseline_onset=0.27,
    baseline_offset=0.28,
    baseline_type='linear',
    baseline_onset_window=500,
    baseline_offset_window=100,
):
    """Poloidal field from raw probe records, filtered before integration.

    Parameters
    ----------
    time : array_like
        Time base [s].
    raw : array_like
        Raw records, one column per probe [V].
    gain : array_like
        Gain per probe [-].
    lowpass_param : Any
        Low-pass specification applied before integration [-].
    baseline_onset : float, optional
        Time at which the signals begin to deviate [s].
    baseline_offset : float, optional
        Time by which they have returned [s].
    baseline_type : str, optional
        Which model to fit as the baseline [-].
    baseline_onset_window : int, optional
        Samples before the onset to include in the fit [-].
    baseline_offset_window : int, optional
        Samples after the offset to include [-].

    Returns
    -------
    tuple of np.ndarray
        The field per probe in tesla, together with the intermediate stages the
        caller may want to inspect [-].

    Processing steps
    ----------------
    1. Apply the per-probe gain.
    2. Low-pass each column.
    3. Integrate in time, carrying the minus sign.
    4. Fit and subtract the two-sided baseline per column.

    Convention
    ----------
    **Gain first, then filter**, which is the order this routine is named for and
    differs from the VEST chain's. The leading minus is Faraday's, applied as the
    caller's system defines it rather than asserted; see this module's conventions.

    Filtering before integration matters: integrating first would accumulate the
    noise the filter is there to remove.

    Defaults
    --------
    The onset, offset and window lengths are machine-specific VEST values.

    Applicability
    -------------
    Machine-independent.  Every machine number is an argument.

    Limitations
    -----------
    Assumes the record is quiet either side of the baseline window. This is the
    routine the fluctuation module points at as the canonical path from pickup
    voltage to field, and a spectrum taken before this step is a derivative
    spectrum, not a field one.

    Provenance
    ----------
    .. [1] The legacy VEST probe chain, kept here in a machine-independent form;
       :mod:`vaft.process.fluctuation` for why the integration must precede a
       spectral index.
    """
    if raw.ndim == 1:
        raw = raw[:, np.newaxis]

    m, n = raw.shape
    if gain.shape[0] != n:
        raise ValueError("Length of gain must match the number of signals (columns in raw).")
    if time.shape[0] != m:
        raise ValueError("Length of time must match number of samples (rows in raw).")

    # Apply gain at the start
    raw = raw * gain

    # Convert baseline onset/offset in seconds to integer indices
    onset_idx = np.searchsorted(time, baseline_onset)
    offset_idx = np.searchsorted(time, baseline_offset)

    baseline_indices = define_baseline(
        time, onset_idx, baseline_onset_window, offset_idx, baseline_offset_window
    )

    # Apply low-pass filter
    filtered_raw = signal.lfilter(lowpass_param, [1.0], raw, axis=0)

    # Integrate to get flux (negative sign if your system defines it so)
    integrated_flux = -cumtrapz_compat(filtered_raw, x=time, initial=0, axis=0)

    # Subtract baseline for each column
    field = np.empty_like(integrated_flux)
    baselines = np.empty_like(integrated_flux)
    for i in range(n):
        flux_corrected, baseline = subtract_baseline(
            time, integrated_flux[:, i], baseline_indices, fitting_opt=baseline_type
        )
        field[:, i] = flux_corrected
        baselines[:, i] = baseline

    return raw, filtered_raw, integrated_flux, field, baselines

def flux_loop_flux(
    time,
    raw,
    gain,
    baseline_onset=0.27,
    baseline_offset=0.28,
    baseline_type='linear',
    baseline_onset_window=500,
    baseline_offset_window=100,
):
    """Poloidal flux from raw flux-loop records, with a two-sided baseline.

    Parameters
    ----------
    time : array_like
        Time base [s].
    raw : array_like
        Raw records, one column per loop [V].
    gain : array_like
        Gain per loop [-].
    baseline_onset : float, optional
        Time at which the signals begin to deviate [s].
    baseline_offset : float, optional
        Time by which they have returned [s].
    baseline_type : str, optional
        Which model to fit as the baseline [-].
    baseline_onset_window : int, optional
        Samples before the onset to include in the fit [-].
    baseline_offset_window : int, optional
        Samples after the offset to include [-].

    Returns
    -------
    np.ndarray
        Flux through each loop, per radian [Wb/rad].

    Processing steps
    ----------------
    1. Apply the per-loop gain.
    2. Integrate in time and divide by two pi.
    3. Fit a baseline over the quiet samples either side of the discharge and
       subtract it per loop.

    Convention
    ----------
    **Per radian**, and the leading minus is Faraday's -- but this is the generic
    routine, so it applies the sign the caller's system uses rather than asserting
    one. The VEST chain asserts it instead; see this module's conventions for why
    the two differ deliberately.

    The baseline is fitted on **both sides** of the discharge, before the onset
    and after the offset, which anchors a drifting integration at both ends rather
    than extrapolating from one.

    Defaults
    --------
    The onset, offset and window lengths are machine-specific VEST values,
    mirroring the discharge timing of that machine. A caller on another machine
    passes its own.

    Applicability
    -------------
    Machine-independent.  Every machine number is an argument.

    Limitations
    -----------
    Assumes the record is genuinely quiet either side of the window given. A
    discharge that has not returned to baseline by the offset biases the fit.

    Provenance
    ----------
    .. [1] The two-sided baseline convention of the legacy VEST workflow, kept
       here in a machine-independent form.
    """
    if raw.ndim == 1:
        raw = raw[:, np.newaxis]

    m, n = raw.shape
    if gain.shape[0] != n:
        raise ValueError("Length of gain must match number of signals.")
    if time.shape[0] != m:
        raise ValueError("Length of time must match number of samples.")

    # Apply gain at the start
    raw = raw * gain

    # Convert baseline onset/offset in seconds to integer indices
    onset_idx = np.searchsorted(time, baseline_onset)
    offset_idx = np.searchsorted(time, baseline_offset)
    baseline_indices = define_baseline(
        time, onset_idx, baseline_onset_window, offset_idx, baseline_offset_window
    )

    # Integrate flux loop data for each signal
    # - sign if that is convention, also / (2*pi)
    integrated_data = -cumtrapz_compat(raw, x=time, initial=0, axis=0) / (2 * np.pi)

    # Remove offset for each signal
    processed_data = np.empty_like(integrated_data)
    baselines = np.empty_like(integrated_data)
    for i in range(n):
        flux_corrected, baseline = subtract_baseline(
            time, integrated_data[:, i], baseline_indices, fitting_opt=baseline_type
        )
        processed_data[:, i] = flux_corrected
        baselines[:, i] = baseline

    return time, processed_data, baselines


# def toroidal_mode_analysis(
#     time_vector, 
#     signal_matrix, 
#     toroidal_angles, 
#     time_points, 
#     window_size=1000, 
#     thres_peak=0.1, 
#     plot_opt=False,
#     nperseg=256,
#     coherence_q=4
# ):
#     """
#     Compute coherence, phase, toroidal mode number, and relative power using the first signal as reference.
    
#     Parameters
#     ----------
#     time_vector : np.ndarray
#         Time axis vector (e.g., 0~1s, 250kHz sampling -> length 250000)
#     signal_matrix : np.ndarray
#         2D array of shape (num_signals x num_samples).
#         Each row represents a different probe(channel), each column represents a time sample.
#     toroidal_angles : np.ndarray
#         Toroidal angles (in radians) corresponding to each probe(row). Length num_signals.
#     time_points : list or np.ndarray
#         Time indices at which to perform analysis (e.g., [1000, 2000, 3000, ...])
#     window_size : int
#         Window size determining how many samples to analyze around each time_point.
#         (default 1000 -> ±500 points)
#     thres_peak : float
#         Minimum height ratio for peak detection relative to maximum spectrum value (default 0.1).
#     plot_opt : bool
#         If True, displays a simple phase plot with slider.
#     nperseg : int
#         nperseg value to use for csd, coherence calculations (default 256).
#     coherence_q : int
#         q value used for coherence threshold calculation. (default 4)
#         Generalizes the original tanh(1.96 / sqrt(2*q-2)) formula.
    
#     Returns
#     -------
#     results : dict
#         {
#           "time": [t1, t2, ...],             # Actual analysis times (seconds)
#           "coherence": [...],               # Array of coherence values for valid peaks for [num_signals-1] channels at each time_point
#           "phase": [...],                   # Array of phase values
#           "mode_number": [...],             # Array of mode numbers
#           "frequencies": [...],             # Array of peak frequencies
#           "power": [...]                    # Array of relative peak powers
#         }
#     """

#     num_signals, num_samples = signal_matrix.shape
#     if len(toroidal_angles) != num_signals:
#         raise ValueError("The number of toroidal angles must match the number of signals.")
    
#     # 샘플링 주파수(Hz)
#     f_sample = 1.0 / np.mean(np.diff(time_vector))
    
#     # 코히런스 임계값(원본 코드 아이디어)
#     coherence_threshold = np.tanh(1.96 / np.sqrt(2 * coherence_q - 2))
    
#     results = {
#         "time": [],
#         "coherence": [],
#         "phase": [],
#         "mode_number": [],
#         "frequencies": [],
#         "power": []
#     }
    
#     all_time_results = []  # 플롯에서 슬라이더로 접근 가능하도록 저장
    
#     half_win = window_size // 2
    
#     for t_idx in time_points:
#         # 창 범위 확인
#         if t_idx < half_win or t_idx >= num_samples - half_win:
#             continue
        
#         window_start = t_idx - half_win
#         window_end   = t_idx + half_win
        
#         ref_signal = signal_matrix[0, window_start:window_end]
#         ref_angle  = toroidal_angles[0]
        
#         time_results = {
#             "coherence": [],
#             "phase": [],
#             "mode_number": [],
#             "frequencies": [],
#             "power": []
#         }
        
#         # 각 프로브(i=1~num_signals-1)에 대해
#         for i in range(1, num_signals):
#             signal_i = signal_matrix[i, window_start:window_end]
            
#             # Cross-spectral density
#             f, pxy = csd(ref_signal, signal_i, fs=f_sample, nperseg=nperseg)
#             magnitude = np.abs(pxy)
            
#             # Coherence
#             _, cxy = coherence(ref_signal, signal_i, fs=f_sample, nperseg=nperseg)
            
#             # 피크 찾기: 크기가 thres_peak * max(magnitude) 이상인 피크
#             peaks, peak_props = find_peaks(
#                 magnitude, 
#                 height=thres_peak * np.max(magnitude)
#             )
#             # 크기 기준 내림차순 정렬
#             peak_heights = peak_props["peak_heights"]
#             desc_order = np.argsort(peak_heights)[::-1]
#             peaks = peaks[desc_order]
            
#             # 코히런스 필터
#             valid_peaks = []
#             for pk in peaks:
#                 if cxy[pk] > coherence_threshold:
#                     valid_peaks.append(pk)
#             valid_peaks = np.array(valid_peaks, dtype=int)
            
#             if len(valid_peaks) > 0:
#                 # 위상(각 유효 피크에서)
#                 phase_vals = np.angle(pxy[valid_peaks])
                
#                 # 모드 번호: (phase / Δphi)
#                 delta_phi = toroidal_angles[i] - ref_angle
#                 n_raw = phase_vals / delta_phi
#                 n_rounded = np.round(n_raw).astype(int)
                
#                 # 상대 파워: 각 피크의 |pxy| / 전체 스펙트럼 |pxy| 합
#                 total_power = np.sum(magnitude)
#                 power_vals  = magnitude[valid_peaks] / total_power
                
#                 time_results["coherence"].append(cxy[valid_peaks])
#                 time_results["phase"].append(phase_vals)
#                 time_results["mode_number"].append(n_rounded)
#                 time_results["frequencies"].append(f[valid_peaks])
#                 time_results["power"].append(power_vals)
#             else:
#                 # 유효 피크가 없으면 빈 배열 저장
#                 time_results["coherence"].append(np.array([]))
#                 time_results["phase"].append(np.array([]))
#                 time_results["mode_number"].append(np.array([]))
#                 time_results["frequencies"].append(np.array([]))
#                 time_results["power"].append(np.array([]))
        
#         # 전체 결과에 추가
#         results["time"].append(time_vector[t_idx])
#         results["coherence"].append(time_results["coherence"])
#         results["phase"].append(time_results["phase"])
#         results["mode_number"].append(time_results["mode_number"])
#         results["frequencies"].append(time_results["frequencies"])
#         results["power"].append(time_results["power"])
        
#         all_time_results.append(time_results)
    
#     if plot_opt:
#         fig, ax = plt.subplots()
#         plt.subplots_adjust(bottom=0.25)

#         def update_plot(idx):
#             ax.clear()
#             time_idx = time_points[idx]
#             time_result = all_time_results[idx]

#             # Plot reference point
#             ax.scatter([toroidal_angles[0]], [0], marker='o', color='red',
#                        label='Reference (0°)', s=100)

#             # Plot phase differences relative to reference
#             for i in range(num_signals - 1):
#                 phases = time_result["phase"][i]
#                 if len(phases) > 0:
#                     # 여러 피크가 있을 수 있으나 여기서는 평균값만 예시로 표시
#                     ax.scatter([toroidal_angles[i+1]], [np.mean(phases)], 
#                                marker='o', label=f'Probe {i+1}')

#             # 대략적 모드번호 피팅 예시 (평균 모드 사용)
#             if any(len(mn) > 0 for mn in time_result["mode_number"]):
#                 valid_modes = [np.mean(mn) for mn in time_result["mode_number"] if len(mn) > 0]
#                 if valid_modes:
#                     mean_mode = np.mean(valid_modes)
#                     theta = np.linspace(0, 2*np.pi, 100)
#                     ax.plot(theta, mean_mode * theta, 'r--', label=f'n={mean_mode:.1f}')

#             ax.set_title(f'Time: {time_vector[time_idx]:.5f} s')
#             ax.set_xlabel('Toroidal Angle (rad)')
#             ax.set_ylabel('Phase Difference (rad)')
#             ax.set_ylim(-np.pi, np.pi)
#             ax.set_xlim(0, 2*np.pi)
#             ax.legend()
#             ax.grid(True)
#             plt.draw()

#         ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
#         slider = Slider(ax_slider, "Time Index", 0, len(time_points) - 1, 
#                         valinit=0, valstep=1)
#         slider.on_changed(lambda val: update_plot(int(val)))
#         update_plot(0)
#         plt.show()

#     return results


