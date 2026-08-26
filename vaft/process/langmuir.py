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
    """Subtract the mean of the first ``n_baseline_samples`` samples."""
    values = np.asarray(data, dtype=float)
    if values.size < int(n_baseline_samples):
        raise ValueError(
            f"remove_offset: need at least {n_baseline_samples} samples, got {values.size}"
        )
    baseline = float(np.mean(values[: int(n_baseline_samples)]))
    return values - baseline


def median_filter_signal(data: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply a one-dimensional median filter."""
    return medfilt(np.asarray(data, dtype=float), kernel_size=int(kernel_size))


def calibrate_voltage(raw: np.ndarray, *, gain: float = 22.0) -> np.ndarray:
    """Convert raw digitized counts/volts to probe voltage: ``V = raw * gain``."""
    return np.asarray(raw, dtype=float) * float(gain)


def calibrate_current(raw: np.ndarray, *, divisor: float = 100.0) -> np.ndarray:
    """Convert raw digitized counts/volts to probe current: ``I = raw / divisor``."""
    if float(divisor) == 0:
        raise ValueError("calibrate_current: divisor must be non-zero")
    return np.asarray(raw, dtype=float) / float(divisor)


def probe_surface_area(*, tip_radius_m: float, tip_length_m: float) -> float:
    """Cylindrical collection area of a probe tip: ``S = 2*pi*r*l``."""
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
    """Solve the classical triple-probe relation for Te at each sample.

    Returns ``(te, solver_ok)``. Samples where no root exists within
    ``te_bounds`` (nonphysical or failed solves) get ``te = nan`` and
    ``solver_ok = False`` rather than a silently substituted value.
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
    """Electron density from the ion-saturation-current triple-probe model.

    ``n_e = sqrt(m_i) * I * exp(1/2) / (S * e * sqrt(e*Te) * (exp(Vd2/Te) - 1))``

    All inputs/outputs are SI: ``current`` in amperes, ``te`` in electron-volts
    (converted internally where the formula needs joules via ``e*Te``),
    ``ion_mass_kg`` in kilograms, ``n_e`` returned in m^-3. ``ion_mass_kg`` is
    required explicitly -- no default gas species is assumed. Nonphysical or
    non-finite results are flagged in the returned mask rather than replaced.
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
    """Run the full triple-probe pipeline on one voltage/current channel pair.

    Verifies (rather than truncates) that ``time_v`` and ``time_i`` agree,
    applies baseline-offset removal and an optional median filter, calibrates
    to physical units, solves for Te, and computes n_e.
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
