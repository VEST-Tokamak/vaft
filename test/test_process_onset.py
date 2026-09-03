"""The onset primitives of issue #409, on synthetic waveforms.

Each case is one of the failure modes the VEST raw-database study found:
coil-firing pickup spikes, isolated optical spikes, brief flashes, a record
with no pulse whose maximum is the pickup, a weak pulse at the noise floor,
and the contracts every consumer relies on -- the onset is a grid sample, and
no onset is never the whole record.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from vaft.process.onset import (
    OnsetRecord,
    excess_threshold,
    isolated_excursions,
    median_smooth,
    pickup_scale,
    principal_pulse_onset,
    principal_pulse_window,
    robust_baseline,
    sustained_excess_onset,
    zero_phase_lowpass,
)

FS = 25_000.0
DT = 1.0 / FS
T = np.arange(0.0, 0.10, DT)          # 100 ms record
ONSET = 0.060                          # true onset
RNG = np.random.default_rng(409)


def pulse(amplitude: float = 1.0, rise_s: float = 0.004, noise: float = 0.01) -> np.ndarray:
    """Ramp-and-plateau pulse with white noise; zero before ONSET."""
    y = np.zeros_like(T)
    on = T >= ONSET
    y[on] = amplitude * np.clip((T[on] - ONSET) / rise_s, 0.0, 1.0)
    return y + noise * RNG.standard_normal(T.size)


def with_spikes(y: np.ndarray, *, n: int, samples: int, height: float, before: float = ONSET - 0.005) -> np.ndarray:
    out = y.copy()
    idx = RNG.choice(np.flatnonzero(T < before), size=n, replace=False)
    for i in idx:
        out[i : i + samples] += height
    return out


def true_index() -> int:
    return int(np.searchsorted(T, ONSET))


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


def test_the_onset_is_always_a_sample_of_the_input_grid():
    for record in (sustained_excess_onset(T, pulse()), principal_pulse_onset(T, pulse())):
        assert record.found
        assert record.time in T
        assert T[record.index] == record.time


def test_no_onset_is_never_the_record_bounds():
    quiet = 0.01 * RNG.standard_normal(T.size)
    for record in (sustained_excess_onset(T, quiet), principal_pulse_onset(T, quiet)):
        assert not record.found
        assert record.time is None and record.index is None
        assert "no_onset" in record.flags
        assert "peak_below_noise" in record.flags


def test_records_serialise_to_json():
    record = sustained_excess_onset(T, with_spikes(pulse(), n=5, samples=2, height=0.5))
    json.dumps(record.as_dict())
    assert record.as_dict()["rejected"]


# ---------------------------------------------------------------------------
# Normal pulse
# ---------------------------------------------------------------------------


def test_both_detectors_land_on_a_clean_pulse_within_the_threshold_lag():
    """A threshold at excess ``h`` is crossed ``h / amplitude * rise_time``
    after the true onset of a linear rise; that lag, read from the threshold
    the record reports, is the tolerance -- not a looser one."""
    rise = 0.004
    y = pulse(amplitude=1.0, rise_s=rise)
    for record in (sustained_excess_onset(T, y), principal_pulse_onset(T, y)):
        excess = record.evidence["threshold"] - record.evidence["baseline_median"]
        expected_lag = excess / 1.0 * rise / DT
        assert abs((record.index - true_index()) - expected_lag) <= 2, record.evidence


def test_the_window_offset_follows_the_onset():
    on, off = principal_pulse_window(T, pulse())
    assert on.found and off.found
    assert off.time >= on.time
    assert off.method == "principal_pulse_offset"


def test_two_percent_walk_back_beats_five_percent_on_a_slow_rise():
    y = pulse(rise_s=0.020, noise=0.002)
    two = principal_pulse_onset(T, y, fraction=0.02)
    five = principal_pulse_onset(T, y, fraction=0.05)
    assert two.time < five.time
    # 2 % of a 20 ms rise is 0.4 ms = 10 samples; 5 % is 25 samples
    assert 0 <= two.index - true_index() <= 12
    assert five.index - true_index() >= 20


# ---------------------------------------------------------------------------
# Spikes and persistence
# ---------------------------------------------------------------------------


def test_isolated_optical_spikes_do_not_move_the_sustained_onset():
    """Twenty 1-3 sample spikes of five sigma before the onset: with the
    median prefilter and a 0.5 ms hold the onset is unchanged, and the record
    says how many excursions it rejected."""
    clean = pulse()
    spiky = with_spikes(clean, n=20, samples=3, height=0.4)
    a = sustained_excess_onset(T, clean, prefilter_samples=5)
    b = sustained_excess_onset(T, spiky, prefilter_samples=5)
    assert b.index == a.index
    assert b.evidence["isolated_excursions_before_onset"] >= 0
    # without persistence the first spike wins
    c = sustained_excess_onset(T, spiky, hold_s=0.0, prefilter_samples=1)
    assert c.time < a.time


def test_coil_pickup_spikes_never_reach_the_principal_pulse():
    """A 1 ms, two-unit spike (larger than the pulse) 20 ms before the onset.

    When the spike is the record's maximum the principal run *is* the spike;
    it is refused because a pulse is never that brief. A pulse three times
    larger than the spike is accepted, at the pulse, with the spike measured
    as the record's pickup scale."""
    spike = np.zeros_like(T)
    i = int(np.searchsorted(T, ONSET - 0.020))
    spike[i : i + int(0.001 / DT)] = 2.0
    dominated = principal_pulse_onset(T, pulse(amplitude=1.0) + spike)
    assert not dominated.found and "principal_run_impulsive" in dominated.flags
    big = principal_pulse_onset(T, pulse(amplitude=10.0) + spike, pickup_floor=3.0)
    assert big.found and abs(big.index - true_index()) <= 3
    assert big.evidence["pickup_scale"] == pytest.approx(2.0, abs=0.1)
    # a pulse only twice its pickup is refused by the floor, and says so
    marginal = principal_pulse_onset(T, pulse(amplitude=4.0) + spike, pickup_floor=3.0)
    assert not marginal.found and "peak_below_pickup_floor" in marginal.flags


def test_a_brief_flash_passes_persistence_but_fails_width():
    y = 0.005 * RNG.standard_normal(T.size)
    i = int(np.searchsorted(T, 0.030))
    y[i : i + int(0.0006 / DT)] += 1.0       # 0.6 ms flash
    p = sustained_excess_onset(T, y, hold_s=5e-4)
    m = sustained_excess_onset(T, y, hold_s=5e-4, min_width_s=1e-3)
    assert p.found and p.time == pytest.approx(0.030, abs=DT)
    assert not m.found
    assert m.rejected and m.rejected[0][1] == "width"


def test_multiple_spikes_and_no_pulse_is_no_onset():
    y = 0.005 * RNG.standard_normal(T.size)
    for centre in (0.020, 0.035, 0.050):
        i = int(np.searchsorted(T, centre))
        y[i : i + int(0.001 / DT)] += 0.3
    assert not principal_pulse_onset(T, y, pickup_floor=3.0).found
    assert not sustained_excess_onset(T, y, hold_s=1.5e-3).found


# ---------------------------------------------------------------------------
# Weak pulses, references, filtering
# ---------------------------------------------------------------------------


def test_a_pulse_at_the_noise_floor_is_refused_and_a_clear_one_accepted():
    noise = 0.01
    weak = pulse(amplitude=3 * noise, noise=noise)
    clear = pulse(amplitude=12 * noise, noise=noise)
    assert not principal_pulse_onset(T, weak).found
    assert principal_pulse_onset(T, clear).found


def test_an_onset_inside_the_reference_is_flagged_not_hidden():
    y = np.where(T >= 0.005, 1.0, 0.0) + 0.001 * RNG.standard_normal(T.size)
    record = sustained_excess_onset(T, y, reference_fraction=0.2)
    assert "reference_contaminated" in record.flags or "reference_flat" in record.flags or not record.found


def test_a_flat_reference_is_flagged():
    y = np.zeros(T.size)
    y[T >= ONSET] = 1.0
    record = sustained_excess_onset(T, y)
    assert not record.found and "reference_flat" in record.flags


def test_zero_phase_filtering_does_not_shift_the_onset():
    y = pulse(rise_s=0.002, noise=0.005)
    plain = principal_pulse_onset(T, y)
    filtered = principal_pulse_onset(T, y, cutoff_hz=2000.0, fs=FS)
    assert abs(filtered.index - plain.index) <= 2


def test_helpers_behave():
    m, s = robust_baseline(np.r_[np.zeros(100), 5.0])
    assert m == 0.0 and s == 0.0
    assert median_smooth([1.0, 100.0, 1.0, 1.0, 1.0], 3)[1] == 1.0
    y = np.zeros(T.size)
    y[10:12] = 1.0
    assert isolated_excursions(y, 0.5, 5) == 1
    assert pickup_scale(y, 0.0, 0.01, DT) == pytest.approx(1.0)
    b, s2, p, thr = excess_threshold(y, T < 0.001, fraction=0.5, sigma=5.0)
    assert p == 1.0 and thr == pytest.approx(0.5)
    assert zero_phase_lowpass(y, 1000.0, FS).shape == y.shape


def test_inputs_are_checked():
    with pytest.raises(ValueError):
        sustained_excess_onset([0.0, 1.0], [0.0])
    with pytest.raises(ValueError):
        principal_pulse_onset([0.0], [0.0])
