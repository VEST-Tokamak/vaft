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
    """A step 5 ms in, with the reference the leading 20 % (20 ms): the
    reference is repaired to its earliest quarter, the onset is found, and
    the record says the reference was contaminated."""
    y = np.where(T >= 0.005, 1.0, 0.0) + 0.001 * RNG.standard_normal(T.size)
    record = sustained_excess_onset(T, y, reference_fraction=0.2)
    assert "reference_contaminated" in record.flags
    assert record.found and record.time == pytest.approx(0.005, abs=2 * DT)


def test_a_transient_inside_the_reference_does_not_refuse_the_pulse():
    """A pre-ionization excursion 10 ms before the pulse, inside a reference
    that was meant to be quiet: the pulse is still found, with the flag."""
    y = pulse(amplitude=100.0, noise=0.1)
    y[(T >= 0.045) & (T < 0.050)] -= 30.0
    record = principal_pulse_onset(T, y, reference_mask=T < 0.055)
    assert record.found and abs(record.index - true_index()) <= 4
    # and one long enough to shift the reference level is repaired, not refused
    y2 = pulse(amplitude=100.0, noise=0.1)
    y2[(T >= 0.040) & (T < 0.055)] -= 30.0
    repaired = principal_pulse_onset(T, y2, reference_mask=T < 0.055)
    assert repaired.found and "reference_contaminated" in repaired.flags


def test_a_noise_dip_during_the_rise_does_not_delay_the_onset():
    y = np.zeros_like(T)
    on = T >= ONSET
    y[on] = np.clip((T[on] - ONSET) / 0.004, 0.0, 1.0)
    y += 0.001 * RNG.standard_normal(T.size)
    clean = sustained_excess_onset(T, y)
    dipped = y.copy()
    dipped[clean.index + 3] = 0.0
    assert sustained_excess_onset(T, dipped).index == clean.index
    assert principal_pulse_onset(T, dipped).index == principal_pulse_onset(T, y).index
    # and a real gap longer than the bridge still splits
    gapped = y.copy()
    gapped[clean.index + 3 : clean.index + 9] = 0.0
    assert sustained_excess_onset(T, gapped).index > clean.index


def test_a_record_too_short_for_the_filter_says_so():
    """A record the filter cannot pad is no evidence, flagged -- not an exception
    (the filter itself still refuses, for callers that reach it directly)."""
    record = principal_pulse_onset(T[:12], np.ones(12), cutoff_hz=2000.0, fs=FS)
    assert not record.found and "record_too_short" in record.flags
    with pytest.raises(ValueError, match="needs more than"):
        zero_phase_lowpass(np.ones(12), 2000.0, FS)


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


# ---------------------------------------------------------------------------
# Windows: onset and offset
# ---------------------------------------------------------------------------

from vaft.process.onset import active_window  # noqa: E402


def box(start: float, end: float, amplitude: float = 1.0, noise: float = 0.005) -> np.ndarray:
    y = np.where((T >= start) & (T < end), amplitude, 0.0)
    return y + noise * RNG.standard_normal(T.size)


def test_a_clean_pulse_has_a_window_from_its_onset_to_its_offset():
    w = active_window(T, box(0.030, 0.070))
    assert w.found and w.duration_s == pytest.approx(0.040, abs=3 * DT)
    assert w.start == pytest.approx(0.030, abs=2 * DT) and w.end == pytest.approx(0.070, abs=2 * DT)
    assert w.start in T and w.end in T and len(w.segments) == 1 and w.flags == ()


def test_a_dip_shorter_than_the_gap_does_not_end_the_window():
    y = box(0.030, 0.070)
    dip = (T >= 0.050) & (T < 0.0506)                 # 0.6 ms below threshold
    y[dip] = 0.0
    w = active_window(T, y, gap_s=1e-3)
    assert len(w.segments) == 1 and w.end == pytest.approx(0.070, abs=2 * DT)
    split = active_window(T, y, gap_s=0.0, post_quiet_s=0.0)
    assert len(split.segments) == 2 and "multiple_segments" in split.flags
    assert split.end == pytest.approx(0.070, abs=2 * DT)   # the envelope still spans both


def test_a_re_emergence_within_the_quiet_time_extends_the_window_and_a_later_one_does_not():
    y = box(0.030, 0.060) + box(0.0615, 0.070, noise=0.0)    # 1.5 ms gap
    w = active_window(T, y, gap_s=0.0, post_quiet_s=2e-3)
    assert len(w.segments) == 1 and w.end == pytest.approx(0.070, abs=2 * DT)
    far = box(0.030, 0.060) + box(0.080, 0.090, noise=0.0)   # 20 ms gap
    w2 = active_window(T, far, gap_s=0.0, post_quiet_s=2e-3)
    assert len(w2.segments) == 2 and "multiple_segments" in w2.flags
    principal = active_window(T, far, principal_only=True)
    assert len(principal.segments) == 1
    assert principal.start == pytest.approx(0.030, abs=2 * DT) and principal.end == pytest.approx(0.060, abs=2 * DT)


def test_a_pulse_still_active_at_the_last_sample_is_flagged_not_truncated_silently():
    w = active_window(T, box(0.030, 1.0))
    assert w.found and "offset_at_record_end" in w.flags and w.end == T[-1]
    early = active_window(T, box(-1.0, 0.050), reference_mask=T > 0.060)
    assert "onset_at_record_start" in early.flags


def test_pre_ionization_light_and_the_main_pulse_are_two_segments_and_principal_picks_the_main():
    y = box(0.020, 0.026, amplitude=0.3) + box(0.040, 0.080, amplitude=1.0, noise=0.0)
    envelope = active_window(T, y)
    assert len(envelope.segments) == 2 and envelope.start == pytest.approx(0.020, abs=2 * DT)
    main = active_window(T, y, principal_only=True)
    assert main.start == pytest.approx(0.040, abs=2 * DT) and main.end == pytest.approx(0.080, abs=2 * DT)


def test_principal_pulse_window_offset_survives_a_brief_dip():
    y = box(0.030, 0.070)
    y[(T >= 0.050) & (T < 0.0506)] = 0.0
    on, off = principal_pulse_window(T, y)
    assert on.found and off.found and off.time == pytest.approx(0.070, abs=2 * DT)
    assert off.method == "principal_pulse_offset"


def test_no_pulse_gives_no_window():
    w = active_window(T, 0.005 * RNG.standard_normal(T.size))
    assert not w.found and "no_onset" in w.flags and w.duration_s is None
    json.dumps(w.as_dict())


def test_a_post_pulse_baseline_shift_does_not_push_the_offset_to_the_record_end():
    """A plasma-current record settles a few percent of its peak above zero
    after the plasma: the offset is judged against that trailing level, not
    the leading one, and the window ends where the current collapsed."""
    y = box(0.030, 0.070, amplitude=80.0, noise=0.15)
    y[T >= 0.070] += 2.5            # 3 % of peak, above the 2 % onset threshold
    w = active_window(T, y, principal_only=True, reference_mask=T < 0.020)
    assert w.found and "offset_at_record_end" not in w.flags
    assert w.end == pytest.approx(0.070, abs=3 * DT)
    assert w.evidence["trailing_quiet"] is True
    # a pulse genuinely active to the end is still reported as such
    still = active_window(T, box(0.030, 1.0, amplitude=80.0, noise=0.15), principal_only=True, reference_mask=T < 0.020)
    assert "offset_at_record_end" in still.flags and still.evidence["trailing_quiet"] is False


def test_a_slow_post_pulse_decay_ends_the_window_at_the_collapse_with_end_fraction():
    """After a termination a plasma-current record decays from ~8 % of peak
    over tens of ms (induced vessel current). With the onset fraction the
    window runs on into that tail; with end_fraction 10 % it ends at the
    collapse."""
    y = box(0.030, 0.060, amplitude=80.0, noise=0.1)
    tail = T >= 0.060
    y[tail] += 6.5 * np.exp(-(T[tail] - 0.060) / 0.030)      # 8 % of peak, 30 ms decay
    long = active_window(T, y, principal_only=True, reference_mask=T < 0.020)
    collapse = active_window(T, y, principal_only=True, reference_mask=T < 0.020, end_fraction=0.10)
    assert long.end > 0.070
    assert collapse.end == pytest.approx(0.060, abs=3 * DT)
    assert collapse.start == long.start


def test_a_disruption_tail_above_the_end_fraction_ends_at_the_last_steep_fall():
    """A disruption collapses the current to a tail at a third of the peak that
    decays for the rest of the record: no level separates it from plasma, the
    last steep fall does. A mid-pulse drop that the plasma survives is not
    the last fall, so it does not end the window."""
    y = np.where((T >= 0.020) & (T < 0.050), 100.0, 0.0)
    y[(T >= 0.035) & (T < 0.050)] = 60.0                        # mid-pulse drop, plasma continues
    tail = T >= 0.050
    y[tail] = 15.0 * np.exp(-(T[tail] - 0.050) / 0.200)        # tail at 15 % of peak, a quarter of the pre-quench current
    y += 0.1 * RNG.standard_normal(T.size)
    w = active_window(T, y, principal_only=True, reference_mask=T < 0.015, end_fraction=0.10)
    assert w.found and "offset_from_collapse" in w.flags and "offset_at_record_end" not in w.flags
    assert w.end == pytest.approx(0.050, abs=5 * DT)
    assert w.evidence["collapse_time"] == pytest.approx(0.050, abs=5 * DT)
    plain = active_window(T, y, principal_only=True, reference_mask=T < 0.015, end_fraction=0.10, collapse_fallback=False)
    assert "offset_at_record_end" in plain.flags


def test_a_record_too_short_for_the_filter_is_no_evidence_not_an_error():
    """Review finding: the zero-phase filter raised before the degenerate checks ran."""
    from vaft.process.onset import active_window

    short = T[:12]
    window = active_window(short, pulse()[:12], cutoff_hz=2000.0)
    assert not window.found
    assert "record_too_short" in window.flags
    record = principal_pulse_onset(short, pulse()[:12], cutoff_hz=2000.0)
    assert not record.found and "record_too_short" in record.flags
    # without a filter the same twelve samples are judged (and found wanting) normally
    assert "record_too_short" not in active_window(short, pulse()[:12]).flags


def test_a_grid_that_cannot_carry_the_cutoff_skips_the_filter_and_says_so():
    """A 2 kHz rule on a 2.85 kHz grid: nothing to remove, nothing to design."""
    from vaft.process.onset import active_window

    coarse = np.linspace(0.0, 0.14, 400)
    y = np.where(coarse >= 0.06, 1.0, 0.0) + 0.01 * RNG.standard_normal(coarse.size)
    window = active_window(coarse, y, cutoff_hz=2000.0, principal_only=True)
    assert window.found
    assert "lowpass_skipped" in window.flags
    fine = active_window(T, pulse(), cutoff_hz=2000.0, principal_only=True)
    assert "lowpass_skipped" not in fine.flags
    # the flag travels with every verdict, not only a found one ...
    quiet = active_window(coarse, 0.01 * RNG.standard_normal(coarse.size), cutoff_hz=2000.0)
    assert not quiet.found and "lowpass_skipped" in quiet.flags
    noise = 0.01 * RNG.standard_normal(coarse.size)
    assert "lowpass_skipped" in principal_pulse_onset(coarse, noise, cutoff_hz=2000.0).flags
    # ... and a record too short for a filter that would not run is judged, not refused
    short = active_window(coarse[:14], y[:14], cutoff_hz=2000.0)
    assert "record_too_short" not in short.flags and "lowpass_skipped" in short.flags


# ---------------------------------------------------------------------------
# Zero crossing after an anchored excursion (#409 PR-v)
# ---------------------------------------------------------------------------

from vaft.process.onset import zero_crossing_after_excursion  # noqa: E402

ANCHOR = 0.030


def excursion(*, depth: float = -5.0, decay_s: float = 0.004, noise: float = 0.02,
              approach: bool = False, cross: bool = True) -> np.ndarray:
    """A loop-voltage-like swing starting at ANCHOR: a sharp dip that decays
    and either crosses zero on a slow positive ramp (``cross``), stalls just
    short of it, or (``approach``, the 41524 shape) comes within a few percent
    of zero, dips again and only then crosses."""
    y = np.zeros_like(T)
    on = T >= ANCHOR
    tau = T[on] - ANCHOR
    swing = depth * np.exp(-tau / decay_s)
    if approach:
        swing += 0.5 * depth * np.exp(-((tau - 0.016) / 0.0015) ** 2)
    if cross:
        swing += 1.0 * np.clip((tau - 0.018) / 0.010, 0.0, 1.0)
    else:
        swing -= 0.4
    y[on] = swing
    return y + noise * RNG.standard_normal(T.size)


def _record(y, **kw):
    return zero_crossing_after_excursion(T, y, anchor_time=ANCHOR, reference_mask=T < ANCHOR - 0.005, **kw)


def test_the_crossing_is_the_first_sign_change_after_the_extremum():
    y = excursion()
    rec = _record(y)
    assert rec.found and rec.method == "excursion_zero_crossing"
    assert rec.time in T
    i = rec.index
    assert rec.evidence["extremum"] < -4.0
    assert rec.evidence["extremum_time"] >= ANCHOR
    assert (y[i - 1] - rec.evidence["baseline_median"]) < 0 < (y[i] - rec.evidence["baseline_median"])
    assert "approached_without_crossing" not in rec.flags
    assert abs(rec.evidence["run_start_minus_anchor"]) <= 5e-4
    json.dumps(rec.as_dict())


def test_an_approach_that_re_dips_before_crossing_is_flagged():
    rec = _record(excursion(approach=True))
    assert rec.found
    assert "approached_without_crossing" in rec.flags
    assert rec.evidence["approach_time"] < rec.time
    assert abs(rec.evidence["approach_min"]) < 0.10 * abs(rec.evidence["extremum"])


def test_a_flat_record_has_no_excursion_at_the_anchor():
    rec = _record(0.02 * RNG.standard_normal(T.size))
    assert not rec.found
    assert "no_excursion_at_anchor" in rec.flags
    assert rec.evidence["anchor_time"] == ANCHOR


def test_a_later_larger_pulse_does_not_steal_the_anchor():
    y = excursion()
    late = T >= 0.080
    y[late] += 12.0 * np.clip((T[late] - 0.080) / 0.003, 0, 1)   # the 41524 plasma-phase swing
    rec = _record(y)
    assert rec.found
    assert rec.evidence["extremum"] < 0 and rec.evidence["extremum_time"] < 0.040
    assert rec.time < 0.080


def test_an_excursion_that_never_crosses_says_so():
    rec = _record(excursion(cross=False))
    assert not rec.found
    assert "no_zero_crossing" in rec.flags
    assert rec.evidence["extremum"] < -4.0


def test_a_run_outside_the_anchor_tolerance_is_rejected():
    y = excursion()
    rec = zero_crossing_after_excursion(T, y, anchor_time=ANCHOR - 0.010,
                                        reference_mask=T < ANCHOR - 0.015, anchor_tolerance_s=2e-3)
    assert not rec.found
    assert "no_excursion_at_anchor" in rec.flags
    assert any(why == "not_at_anchor" for _, why, _ in rec.rejected)


def test_a_sample_exactly_at_the_baseline_is_not_a_crossing():
    y = np.resize([0.0, 1e-3, -1e-3], T.size)   # reference median exactly 0, spread finite
    i0 = int(np.searchsorted(T, ANCHOR))
    y[i0:i0 + 40] = -5.0
    y[i0 + 40:i0 + 50] = 0.0          # sits at the baseline
    y[i0 + 50:] = 2.0                 # crosses here
    rec = zero_crossing_after_excursion(T, y, anchor_time=ANCHOR, reference_mask=T < ANCHOR - 0.005,
                                        sigma=0.0, fraction=0.05)
    assert rec.index == i0 + 50
