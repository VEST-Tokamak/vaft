"""Unit tests for toroidal mode analysis and phase fitting sign conventions (Issue #638).

Verifies that:
1. `toroidal_mode_analysis` and `toroidal_phase_fit_at_time` yield identical mode
   numbers for identical synthetic signals across positive, negative, and zero modes.
2. Positive n corresponds to co-current / +phi propagation (downstream phase lag).
3. Negative n corresponds to counter-current / -phi propagation (downstream phase lead).
4. Geometry sign inversions (Delta_phi < 0) are handled consistently.
5. Edge cases including phase wrapping and multiple simultaneous modes are recovered.
"""

from __future__ import annotations

import numpy as np
import pytest

from vaft.process.magnetics import (
    toroidal_mode_analysis,
    toroidal_phase_fit_at_time,
)


@pytest.mark.parametrize("n_expected", [1, 2, -1, -2, 0])
def test_toroidal_mode_analysis_and_fit_agree(n_expected: int):
    """Both entry points must yield the exact same mode number for identical signals."""
    sample_rate = 50_000.0
    n_samples = 4096
    time = np.arange(n_samples, dtype=float) / sample_rate
    frequency = 5_000.0
    initial_phase = 0.35

    # 4 sensors spaced around the torus
    angles = np.deg2rad([0.0, 30.0, 60.0, 90.0])
    phase_geometry = float(angles[1] - angles[0])

    # Under canonical convention: phase(phi) = initial_phase - n * phi
    phases = initial_phase - float(n_expected) * angles
    signals = np.vstack(
        [np.sin(2.0 * np.pi * frequency * time + phi) for phi in phases]
    )

    # 1. Pair analysis on probe 0 and probe 1
    res_pair = toroidal_mode_analysis(
        signals[0],
        signals[1],
        sample_rate=sample_rate,
        phase_geometry=phase_geometry,
        peak_threshold=0.05,
        nperseg=1024,
    )
    assert len(res_pair.n) > 0, "Expected at least one coherent peak in pair analysis"
    # Find peak closest to target frequency
    freq_idx = int(np.argmin(np.abs(res_pair.frequency - frequency)))
    n_pair = int(res_pair.n[freq_idx])

    # 2. Wrapped phase array fit across all 4 probes
    res_fit = toroidal_phase_fit_at_time(
        time,
        signals,
        angles,
        center_time=0.04,
        sample_rate=sample_rate,
        window_size=512,
        frequencies=[frequency],
        candidate_n=tuple(range(-4, 5)),
    )
    assert len(res_fit.modes) == 1, "Expected one fitted mode"
    n_fit = res_fit.modes[0].n

    # Assert both routines match the expected physical mode number and each other
    assert n_pair == n_expected, f"Pair analysis got n={n_pair}, expected {n_expected}"
    assert n_fit == n_expected, f"Array fit got n={n_fit}, expected {n_expected}"
    assert n_pair == n_fit, f"Pair analysis (n={n_pair}) != Array fit (n={n_fit})"


def test_co_current_and_counter_current_propagation():
    """Verify co-current (+phi) is n > 0 and counter-current (-phi) is n < 0."""
    sample_rate = 100_000.0
    time = np.arange(4096, dtype=float) / sample_rate
    freq = 8_000.0
    dphi = np.pi / 4  # 45 degrees

    # Co-current wave propagating in +phi: downstream probe lags upstream probe
    # signal_upstream = sin(omega * t)
    # signal_downstream = sin(omega * t - delta) with delta > 0
    sig_up = np.sin(2.0 * np.pi * freq * time)
    sig_down_lag = np.sin(2.0 * np.pi * freq * time - 2 * dphi)  # n = +2
    res_co = toroidal_mode_analysis(
        sig_up,
        sig_down_lag,
        sample_rate=sample_rate,
        phase_geometry=dphi,
        peak_threshold=0.05,
        nperseg=1024,
    )
    assert 2 in set(res_co.n.astype(int))

    # Counter-current wave propagating in -phi: downstream probe leads upstream probe
    sig_down_lead = np.sin(2.0 * np.pi * freq * time + 2 * dphi)  # n = -2
    res_counter = toroidal_mode_analysis(
        sig_up,
        sig_down_lead,
        sample_rate=sample_rate,
        phase_geometry=dphi,
        peak_threshold=0.05,
        nperseg=1024,
    )
    assert -2 in set(res_counter.n.astype(int))


def test_negative_phase_geometry():
    """When coil B is at smaller toroidal angle than coil A (dphi < 0), n is preserved."""
    sample_rate = 50_000.0
    time = np.arange(4096, dtype=float) / sample_rate
    freq = 6_000.0
    phi_a = np.pi / 3
    phi_b = 0.0
    dphi = phi_b - phi_a  # -pi/3 < 0
    expected_n = 2

    sig_a = np.sin(2.0 * np.pi * freq * time - expected_n * phi_a)
    sig_b = np.sin(2.0 * np.pi * freq * time - expected_n * phi_b)

    res = toroidal_mode_analysis(
        sig_a,
        sig_b,
        sample_rate=sample_rate,
        phase_geometry=dphi,
        peak_threshold=0.05,
        nperseg=1024,
    )
    assert expected_n in set(res.n.astype(int))


def test_multiple_simultaneous_modes():
    """Verify both routines on a composite signal containing two distinct mode frequencies."""
    sample_rate = 100_000.0
    time = np.arange(8192, dtype=float) / sample_rate
    f1, n1 = 4_000.0, 1
    f2, n2 = 12_000.0, 3

    angles = np.deg2rad([0.0, 30.0, 60.0, 90.0])
    dphi = float(angles[1] - angles[0])

    signals = np.vstack(
        [
            np.sin(2.0 * np.pi * f1 * time - n1 * phi)
            + 0.8 * np.sin(2.0 * np.pi * f2 * time - n2 * phi)
            for phi in angles
        ]
    )

    # Pair analysis
    res_pair = toroidal_mode_analysis(
        signals[0],
        signals[1],
        sample_rate=sample_rate,
        phase_geometry=dphi,
        peak_threshold=0.05,
        nperseg=2048,
    )
    # Check that both n1 and n2 are present at their respective frequencies
    idx1 = int(np.argmin(np.abs(res_pair.frequency - f1)))
    idx2 = int(np.argmin(np.abs(res_pair.frequency - f2)))
    assert int(res_pair.n[idx1]) == n1
    assert int(res_pair.n[idx2]) == n2

    # Array fit
    res_fit = toroidal_phase_fit_at_time(
        time,
        signals,
        angles,
        center_time=0.04,
        sample_rate=sample_rate,
        window_size=1024,
        frequencies=[f1, f2],
        candidate_n=tuple(range(-4, 5)),
    )
    fit_modes = {round(m.frequency, -2): m.n for m in res_fit.modes}
    assert fit_modes[round(f1, -2)] == n1
    assert fit_modes[round(f2, -2)] == n2
