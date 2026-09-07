"""Backend-independent VEST triple Langmuir probe physics.

This module is machine-independent: raw VEST field selection, shot-era bias
voltage/tip geometry, and ODS population belong in
``vaft.machine_mapping.langmuir_probes``. Callers here pass already-selected
physical signals (or raw counts plus explicit calibration factors) and get
back electron temperature/density.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import brentq
from scipy.signal import medfilt

ELEMENTARY_CHARGE_C = 1.602176634e-19

__all__ = [
    "remove_offset",
    "median_filter_signal",
    "calibrate_voltage",
    "calibrate_current",
    "probe_surface_area",
    "solve_electron_temperature",
    "electron_density",
    "process_triple_probe",
]


def remove_offset(data: np.ndarray, *, n_baseline_samples: int = 500) -> np.ndarray:
    """Subtract a signal's pre-plasma baseline, taken as its leading samples.

    Parameters
    ----------
    data : array_like
        The waveform, in whatever unit it has [any].
    n_baseline_samples : int, optional
        How many leading samples make up the baseline [-].

    Returns
    -------
    np.ndarray
        The waveform with its baseline mean removed [any].

    Raises
    ------
    ValueError
        The record is shorter than the requested baseline.

    Defaults
    --------
    ``n_baseline_samples = 500`` is an acquisition-era value carried from the VEST
    probe digitizer settings, where the leading samples precede the discharge. It
    is a sample count, not a duration, so it means a different interval on a
    different sample rate.

    Convention
    ----------
    The mean of the leading window, not a fitted trend, so a baseline that drifts
    during the shot is not removed.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Assumes the leading window is signal-free. A probe already collecting current
    at the start of the record has its baseline biased by that current.

    Provenance
    ----------
    .. [1] ``vest.yaml`` ``langmuir_probes`` processing, which carries the VEST
       sample count this default mirrors.
    """
    values = np.asarray(data, dtype=float)
    if values.size < int(n_baseline_samples):
        raise ValueError(
            f"remove_offset: need at least {n_baseline_samples} samples, got {values.size}"
        )
    baseline = float(np.mean(values[: int(n_baseline_samples)]))
    return values - baseline


def median_filter_signal(data: np.ndarray, kernel_size: int) -> np.ndarray:
    """Median-filter a waveform along its only axis.

    Parameters
    ----------
    data : array_like
        The waveform [any].
    kernel_size : int
        Width of the median window in samples; must be odd [-].

    Returns
    -------
    np.ndarray
        The filtered waveform, same length [any].

    Convention
    ----------
    A median filter is zero phase: it moves no feature in time, unlike a causal
    low-pass. It removes isolated spikes without the smoothing a mean filter
    applies to a real edge, which is why a probe trace uses it.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    The kernel must be odd; SciPy raises otherwise. Edges are handled by zero
    padding, so the first and last half-kernel of samples are pulled toward zero.
    A feature narrower than half the kernel is removed whether or not it is noise.

    Provenance
    ----------
    .. [1] ``scipy.signal.medfilt``. Issue #152 does not record the kernel size the
       original workflow used, which is why the pipeline leaves this unset rather
       than guessing one; see :func:`process_triple_probe`.
    """
    return medfilt(np.asarray(data, dtype=float), kernel_size=int(kernel_size))


def calibrate_voltage(raw: np.ndarray, *, gain: float = 22.0) -> np.ndarray:
    """Convert a raw probe voltage record to physical volts.

    Parameters
    ----------
    raw : array_like
        Digitized probe signal [V].
    gain : float, optional
        Multiplicative divider gain [-].

    Returns
    -------
    np.ndarray
        Probe voltage [V].

    Defaults
    --------
    ``gain = 22.0`` is machine-specific: it is the VEST probe divider, mirrored
    from ``vest.yaml``. This module is otherwise machine-independent, so a caller
    on another machine must pass its own value rather than inherit this one.

    Applicability
    -------------
    Machine-independent.  The arithmetic is a scale factor; only the default
    carries a VEST number, and a caller on another machine passes its own.

    Provenance
    ----------
    .. [1] ``vest.yaml`` ``langmuir_probes`` voltage gain.
    """
    return np.asarray(raw, dtype=float) * float(gain)


def calibrate_current(raw: np.ndarray, *, divisor: float = 100.0) -> np.ndarray:
    """Convert a raw probe current record to amperes.

    Parameters
    ----------
    raw : array_like
        Digitized probe signal [V].
    divisor : float, optional
        Shunt or amplifier divisor, non-zero [-].

    Returns
    -------
    np.ndarray
        Probe current [A].

    Raises
    ------
    ValueError
        The divisor is zero.

    Defaults
    --------
    ``divisor = 100.0`` is machine-specific, the VEST current divisor mirrored
    from ``vest.yaml``, and carries the same caveat as the voltage gain: a caller
    on another machine must supply its own.

    Applicability
    -------------
    Machine-independent.  The arithmetic is a scale factor; only the default
    carries a VEST number, and a caller on another machine passes its own.

    Provenance
    ----------
    .. [1] ``vest.yaml`` ``langmuir_probes`` current divisor.
    """
    if float(divisor) == 0:
        raise ValueError("calibrate_current: divisor must be non-zero")
    return np.asarray(raw, dtype=float) / float(divisor)


def probe_surface_area(*, tip_radius_m: float, tip_length_m: float) -> float:
    """Collection area of a cylindrical probe tip.

    Parameters
    ----------
    tip_radius_m : float
        Tip radius, strictly positive [m].
    tip_length_m : float
        Exposed tip length, strictly positive [m].

    Returns
    -------
    float
        The collecting surface area [m^2].

    Raises
    ------
    ValueError
        Either dimension is not positive.

    Convention
    ----------
    The lateral area of a cylinder, ``2*pi*r*l``. The end cap is excluded, which
    is the usual convention for a probe whose length far exceeds its radius, and
    the sheath is assumed thin enough that the collecting area is the geometric
    one.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    A thick sheath at low density collects over an area larger than the geometric
    one, which this does not model; the density inferred from it is then an
    overestimate.

    Provenance
    ----------
    .. [1] The standard cylindrical-probe collection area used by the
       triple-probe density relation in :func:`electron_density`.
    """
    if tip_radius_m <= 0 or tip_length_m <= 0:
        raise ValueError("probe_surface_area: tip_radius_m and tip_length_m must be positive")
    return 2.0 * np.pi * float(tip_radius_m) * float(tip_length_m)


def _triple_probe_residual(te: float, vd2: float, vd3: float) -> float:
    # Classical triple-probe relation (theoretical form):
    #   (1 - exp(-Vd2/Te)) / (1 - exp(-Vd3/Te)) = 1/2
    # The MATLAB expression quoted by issue #152 parses as
    #   1 - exp(-Vd2/Te)/(1 - exp(-Vd3/Te)) - 1/2 = 0
    # which is not the same equation (a `1 - a/b` transcription of `(1-a)/b`).
    # No reference .mat data was available in-repo to arbitrate between the two
    # (see issue #152 discussion), so this implements the theoretically correct
    # relation rather than reproducing the apparent transcription bug.
    denominator = 1.0 - np.exp(-vd3 / te)
    numerator = 1.0 - np.exp(-vd2 / te)
    return numerator / denominator - 0.5


def solve_electron_temperature(
    vd2: np.ndarray,
    vd3: np.ndarray,
    *,
    te_bounds: tuple[float, float] = (0.1, 500.0),
    rtol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the classical triple-probe relation for the electron temperature.

    Parameters
    ----------
    vd2 : array_like
        Voltage difference between the first probe pair [V].
    vd3 : array_like
        Voltage difference between the second pair, same shape [V].
    te_bounds : tuple of float, optional
        Bracket the root is searched in [eV].
    rtol : float, optional
        Relative tolerance of the root finder [-].

    Returns
    -------
    te : np.ndarray
        Electron temperature per sample, NaN where no root was found [eV].
    solver_ok : np.ndarray
        ``True`` where a root was found inside the bracket [-].

    Raises
    ------
    ValueError
        The two voltage arrays do not have the same shape.

    Convention
    ----------
    Solves ``(1 - exp(-Vd2/Te)) / (1 - exp(-Vd3/Te)) = 1/2``, the theoretical
    triple-probe relation, sample by sample by bracketed root finding. The one
    half on the right is the definition of the technique, not a tunable.

    Defaults
    --------
    The bracket and the tolerance are numerical conveniences: the bracket spans
    the range a probe measurement can plausibly occupy, and the tolerance is far
    below the scatter of the voltages that feed it.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    A sample with no root inside the bracket yields NaN and a false flag rather
    than a substituted value, so a caller must honour the mask. The relation
    assumes a Maxwellian electron distribution, equal collecting areas on the
    three tips, and a thin sheath.

    The implemented relation is the theoretically correct one. The MATLAB
    expression quoted by issue #152 parses as a different equation, a
    ``1 - a/b`` transcription of ``(1-a)/b``, and no reference data was available
    to arbitrate; this deliberately does not reproduce that apparent transcription
    error.

    Provenance
    ----------
    .. [1] The classical triple-probe relation; the transcription discrepancy is
       recorded in issue #152.
    """
    vd2_array = np.asarray(vd2, dtype=float).reshape(-1)
    vd3_array = np.asarray(vd3, dtype=float).reshape(-1)
    if vd2_array.shape != vd3_array.shape:
        raise ValueError(
            f"solve_electron_temperature: vd2 and vd3 must have the same shape "
            f"({vd2_array.shape} != {vd3_array.shape})"
        )

    te = np.full(vd2_array.shape, np.nan, dtype=float)
    solver_ok = np.zeros(vd2_array.shape, dtype=bool)
    lower, upper = float(te_bounds[0]), float(te_bounds[1])

    for index in range(vd2_array.size):
        v2 = vd2_array[index]
        v3 = vd3_array[index]
        try:
            residual_lower = _triple_probe_residual(lower, v2, v3)
            residual_upper = _triple_probe_residual(upper, v2, v3)
            if not np.isfinite(residual_lower) or not np.isfinite(residual_upper):
                continue
            if residual_lower * residual_upper > 0:
                continue
            root = brentq(_triple_probe_residual, lower, upper, args=(v2, v3), rtol=rtol)
        except (ValueError, ZeroDivisionError, FloatingPointError):
            continue
        te[index] = root
        solver_ok[index] = True

    return te, solver_ok


def electron_density(
    vd2: np.ndarray,
    te: np.ndarray,
    current: np.ndarray,
    *,
    tip_radius_m: float,
    tip_length_m: float,
    ion_mass_kg: float,
    e: float = ELEMENTARY_CHARGE_C,
) -> tuple[np.ndarray, np.ndarray]:
    """Electron density from the ion saturation current and the solved temperature.

    Parameters
    ----------
    vd2 : array_like
        Voltage difference between the first probe pair [V].
    te : array_like
        Electron temperature per sample, same shape [eV].
    current : array_like
        Probe current, same shape [A].
    tip_radius_m : float
        Tip radius [m].
    tip_length_m : float
        Exposed tip length [m].
    ion_mass_kg : float
        Ion mass; required, with no assumed species [kg].
    e : float, optional
        Elementary charge [C].

    Returns
    -------
    n_e : np.ndarray
        Electron density, NaN where non-physical [m^-3].
    valid : np.ndarray
        ``True`` where the density is finite and positive [-].

    Raises
    ------
    ValueError
        The ion mass is not positive, or the three arrays differ in shape.

    Convention
    ----------
    ``n_e = sqrt(m_i) * I * exp(1/2) / (S * e * sqrt(e*Te) * (exp(Vd2/Te) - 1))``.
    Everything is SI except the temperature, which is in electron-volts and is
    converted where the formula needs joules. **The ion mass has no default**: the
    density scales as its square root, so assuming a species would put a silent
    factor into every result.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Non-physical or non-finite results are flagged rather than replaced, so the
    mask must be honoured. Inherits the assumptions of
    :func:`solve_electron_temperature` and the thin-sheath geometric area of
    :func:`probe_surface_area`.

    Provenance
    ----------
    .. [1] The ion-saturation-current form of the triple-probe density relation;
       see issue #152 for the port's history.
    """
    if ion_mass_kg <= 0:
        raise ValueError("electron_density: ion_mass_kg must be positive")

    vd2_array = np.asarray(vd2, dtype=float).reshape(-1)
    te_array = np.asarray(te, dtype=float).reshape(-1)
    current_array = np.asarray(current, dtype=float).reshape(-1)
    if not (vd2_array.shape == te_array.shape == current_array.shape):
        raise ValueError(
            "electron_density: vd2, te, and current must share the same shape "
            f"({vd2_array.shape}, {te_array.shape}, {current_array.shape})"
        )

    surface_area = probe_surface_area(tip_radius_m=tip_radius_m, tip_length_m=tip_length_m)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        denominator = surface_area * e * np.sqrt(e * te_array) * (np.exp(vd2_array / te_array) - 1.0)
        n_e = np.sqrt(ion_mass_kg) * current_array * np.exp(0.5) / denominator

    valid = np.isfinite(n_e) & (n_e > 0)
    n_e = np.where(valid, n_e, np.nan)
    return n_e, valid


def process_triple_probe(
    time_v: np.ndarray,
    vd2_raw: np.ndarray,
    time_i: np.ndarray,
    i_raw: np.ndarray,
    vd3: float,
    *,
    tip_radius_m: float,
    tip_length_m: float,
    ion_mass_kg: float,
    voltage_gain: float = 22.0,
    current_divisor: float = 100.0,
    n_baseline_samples: int = 500,
    median_kernel: int | None = None,
    time_rtol: float = 1e-6,
    time_atol: float = 1e-9,
) -> dict[str, Any]:
    """Run the full triple-probe chain on one voltage and current channel pair.

    Parameters
    ----------
    time_v : array_like
        Time base of the voltage record [s].
    vd2_raw : array_like
        Raw voltage record [V].
    time_i : array_like
        Time base of the current record; must agree with *time_v* [s].
    i_raw : array_like
        Raw current record [V].
    vd3 : float
        The second pair's voltage difference, constant over the record [V].
    tip_radius_m : float
        Tip radius [m].
    tip_length_m : float
        Exposed tip length [m].
    ion_mass_kg : float
        Ion mass; required [kg].
    voltage_gain : float, optional
        Divider gain for the voltage channel [-].
    current_divisor : float, optional
        Divisor for the current channel [-].
    n_baseline_samples : int, optional
        Leading samples forming the baseline [-].
    median_kernel : int, optional
        Median filter width; no filtering when unset [-].
    time_rtol : float, optional
        Relative tolerance on the time-base comparison [-].
    time_atol : float, optional
        Absolute tolerance on that comparison [s].

    Returns
    -------
    dict of str to np.ndarray
        ``time`` in seconds, ``vd2`` in volts, ``current`` in amperes, ``te`` in
        electron-volts, ``n_e`` in inverse cubic metres, and ``solver_ok`` as the
        combined validity mask [-].

    Raises
    ------
    ValueError
        The two time bases do not agree.

    Processing steps
    ----------------
    1. Verify that the voltage and current time bases agree, and refuse if not.
    2. Remove the leading-sample baseline from both records.
    3. Optionally median-filter both.
    4. Calibrate to volts and amperes.
    5. Solve the triple-probe relation for the temperature.
    6. Compute the density, and combine the two validity masks.

    Defaults
    --------
    The gain, divisor and baseline length are machine-specific VEST values
    mirrored from ``vest.yaml``. ``median_kernel`` is deliberately unset:
    issue #152 does not record the kernel the original workflow used, so no
    filtering is applied rather than a guessed width. The time tolerances are
    numerical conveniences.

    Convention
    ----------
    Mismatched time bases are an error, not something to fix by truncating to the
    shorter record; index-truncating two differently sampled channels silently
    misaligns them. The second pair's voltage is a scalar broadcast over the
    record, since it is a fixed bias rather than a measurement.

    Applicability
    -------------
    Machine-independent.  The chain is generic; the calibration, baseline and
    geometry arguments carry VEST defaults that another machine overrides.

    Limitations
    -----------
    Inherits every assumption of the two inference steps. The returned mask is the
    conjunction of the temperature and density masks, so a sample invalid in
    either is invalid in the result.

    Provenance
    ----------
    .. [1] Issue #152, the port of the VEST triple-probe workflow that this chain
       reproduces, including the decisions it left unresolved.
    """
    time_v_array = np.asarray(time_v, dtype=float).reshape(-1)
    time_i_array = np.asarray(time_i, dtype=float).reshape(-1)
    if time_v_array.shape != time_i_array.shape or not np.allclose(
        time_v_array, time_i_array, rtol=time_rtol, atol=time_atol
    ):
        raise ValueError(
            "process_triple_probe: voltage and current time coordinates are not "
            "aligned; verify or explicitly resample the raw channels rather than "
            "truncating by index"
        )

    voltage_raw = remove_offset(vd2_raw, n_baseline_samples=n_baseline_samples)
    current_raw = remove_offset(i_raw, n_baseline_samples=n_baseline_samples)
    if median_kernel:
        voltage_raw = median_filter_signal(voltage_raw, median_kernel)
        current_raw = median_filter_signal(current_raw, median_kernel)

    vd2 = calibrate_voltage(voltage_raw, gain=voltage_gain)
    current = calibrate_current(current_raw, divisor=current_divisor)

    vd3_array = np.full(vd2.shape, float(vd3), dtype=float)
    te, te_ok = solve_electron_temperature(vd2, vd3_array)
    n_e, n_e_ok = electron_density(
        vd2,
        te,
        current,
        tip_radius_m=tip_radius_m,
        tip_length_m=tip_length_m,
        ion_mass_kg=ion_mass_kg,
    )
    solver_ok = te_ok & n_e_ok

    return {
        "time": time_v_array,
        "vd2": vd2,
        "current": current,
        "te": te,
        "n_e": n_e,
        "solver_ok": solver_ok,
    }
