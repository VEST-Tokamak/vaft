"""PF-current shot-era gains and PF6 saturation repair (issue #195)."""

import gzip
import json

import numpy as np
import pytest

from vaft.machine_mapping.pf_active import _coil_gain_by_index, vfit_pf
from vaft.process.signal_processing import SignalRepairError, repair_clipped_interval

# `coil_gains` keys are 0-based hardware coil indices: 0 = PF1 ... 9 = PF10.
PF1_INDEX = 0
PF2_INDEX = 1
PF5_INDEX = 4
PF6_INDEX = 5


def _write_raw_dump(path, shot, fields):
    payload = {
        "shot": shot,
        "fields": {
            str(field): {
                "data": values.tolist() if hasattr(values, "tolist") else values,
                "type": "slow",
            }
            for field, values in fields.items()
        },
    }
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)


# --------------------------------------------------------------------------
# Shot-era gain boundaries
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("shot", "expected"),
    [
        (45964, -5.0e4),
        (45965, -1.0e4),
    ],
)
def test_pf1_gain_boundary_45965_is_preserved(shot, expected):
    """Regression protection for behavior that was already correct on
    develop, so inserting the new 48372 revision cannot disturb it."""
    assert _coil_gain_by_index(shot)[PF1_INDEX] == pytest.approx(expected)


@pytest.mark.parametrize(
    ("shot", "expected"),
    [
        (38109, 1.0e4),
        (38110, -1.0e4),
        (48371, -1.0e4),
        (48372, -5.0e3),
    ],
)
def test_pf5_gain_history_including_the_new_48372_boundary(shot, expected):
    assert _coil_gain_by_index(shot)[PF5_INDEX] == pytest.approx(expected)


@pytest.mark.parametrize("shot", [45964, 45965, 48371, 48372])
def test_unrelated_coil_gains_are_unchanged_across_the_new_boundary(shot):
    gains = _coil_gain_by_index(shot)
    assert gains[PF2_INDEX] == pytest.approx(1.0e3)
    assert gains[PF6_INDEX] == pytest.approx(-1.0e3)


# --------------------------------------------------------------------------
# Generic clipped-interval repair primitive
# --------------------------------------------------------------------------


def _time(n: int) -> np.ndarray:
    return np.arange(n, dtype=float) * 4e-5


def test_unclipped_waveform_is_returned_unchanged():
    time = _time(200)
    values = np.sin(np.linspace(0.0, 6.0, time.size)) * 100.0
    repaired = repair_clipped_interval(time, values, clip_value=-5000.0, tolerance=10.0)
    np.testing.assert_array_equal(repaired, values)


def test_interior_clip_is_reconstructed_and_other_samples_preserved():
    time = _time(400)
    truth = -4000.0 + 2000.0 * np.sin(np.linspace(0.0, 3.0, time.size))
    clipped = np.maximum(truth, -5000.0)
    # Force an unambiguous saturated plateau in the interior.
    clipped[150:200] = -5000.0

    repaired = repair_clipped_interval(time, clipped, clip_value=-5000.0, tolerance=10.0)

    untouched = np.ones(time.size, dtype=bool)
    untouched[150:200] = False
    np.testing.assert_array_equal(repaired[untouched], clipped[untouched])
    # The reconstruction must actually leave the clip level, not echo it.
    assert np.all(repaired[150:200] < -5000.0 + 1e-9) or np.any(repaired[150:200] != -5000.0)
    assert not np.allclose(repaired[150:200], -5000.0)


def test_smooth_signal_reconstruction_is_close_to_the_underlying_truth():
    time = _time(500)
    truth = -3000.0 - 2500.0 * np.sin(np.linspace(0.0, np.pi, time.size))
    clipped = np.maximum(truth, -5000.0)
    saturated = np.abs(clipped - (-5000.0)) < 10.0
    assert saturated.any()

    repaired = repair_clipped_interval(time, clipped, clip_value=-5000.0, tolerance=10.0)

    # A cubic spline through a smooth sinusoid's shoulders recovers the
    # clipped arc to within a few percent of its depth.
    assert np.max(np.abs(repaired[saturated] - truth[saturated])) < 0.05 * 2500.0


def test_fully_saturated_waveform_raises_instead_of_fabricating():
    time = _time(100)
    values = np.full(time.size, -5000.0)
    with pytest.raises(SignalRepairError, match="no unsaturated support"):
        repair_clipped_interval(time, values, clip_value=-5000.0, tolerance=10.0)


def test_insufficient_support_raises():
    time = _time(100)
    values = np.full(time.size, -5000.0)
    values[10] = -100.0
    values[20] = -120.0
    with pytest.raises(SignalRepairError, match="at least"):
        repair_clipped_interval(time, values, clip_value=-5000.0, tolerance=10.0)


@pytest.mark.parametrize("edge", ["start", "end"])
def test_saturation_touching_a_boundary_raises_rather_than_extrapolating(edge):
    time = _time(200)
    values = -1000.0 + np.linspace(0.0, 500.0, time.size)
    if edge == "start":
        values[:30] = -5000.0
    else:
        values[-30:] = -5000.0
    with pytest.raises(SignalRepairError, match="start or end"):
        repair_clipped_interval(time, values, clip_value=-5000.0, tolerance=10.0)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_input_raises(bad):
    time = _time(200)
    values = -1000.0 + np.linspace(0.0, 500.0, time.size)
    values[100:130] = -5000.0
    values[5] = bad
    with pytest.raises(SignalRepairError, match="non-finite"):
        repair_clipped_interval(time, values, clip_value=-5000.0, tolerance=10.0)


def test_tolerance_band_defines_saturation():
    """Samples just outside the +/-10 A band are measurements, not clipping."""
    time = _time(200)
    values = -1000.0 + np.linspace(0.0, 500.0, time.size)
    values[100:130] = -4985.0  # 15 A away from the limit -> not saturated
    repaired = repair_clipped_interval(time, values, clip_value=-5000.0, tolerance=10.0)
    np.testing.assert_array_equal(repaired, values)


# --------------------------------------------------------------------------
# PF6 wiring
# --------------------------------------------------------------------------


def test_every_acquired_coil_keeps_a_measured_waveform(tmp_path):
    """No acquired coil may be zero-filled. PF2 is absent from Coil_info.mat
    (an explicitly disabled hardware channel, hence a meaningful zero) --
    that is a different thing from the #44 zero-fill defect, and from VFIT's
    solver-side PF2 residual exclusion, which belongs in the inversion layer
    and must never reach the canonical mapping (#195)."""
    shot = 45965
    source = tmp_path / "raw.json.gz"
    samples = np.sin(np.linspace(0.0, 20.0, 1200)) + np.linspace(0.0, 0.2, 1200)
    _write_raw_dump(source, shot, {field: samples for field in (5, 59, 62, 65)})

    _time_axis, currents = vfit_pf(shot, raw_source=source)

    acquired_indices = (PF1_INDEX, PF5_INDEX, PF6_INDEX, 8, 9)
    for coil_index in acquired_indices:
        assert np.any(currents[coil_index] != 0.0), f"coil {coil_index} was zero-filled"
    assert np.all(currents[PF2_INDEX] == 0.0)


def test_unclipped_pf6_waveform_is_not_altered_by_the_repair_hook(tmp_path):
    """The saturation-repair config must be a no-op when nothing is clipped."""
    shot = 45965
    source = tmp_path / "raw.json.gz"
    samples = np.sin(np.linspace(0.0, 20.0, 1200)) + np.linspace(0.0, 0.2, 1200)
    _write_raw_dump(source, shot, {field: samples for field in (5, 59, 62, 65)})

    _time_axis, currents = vfit_pf(shot, raw_source=source)

    pf6 = currents[PF6_INDEX]
    assert np.all(np.isfinite(pf6))
    assert np.min(np.abs(pf6 - (-5000.0))) > 10.0  # nothing near the clip level


def test_clipped_pf6_waveform_is_repaired_end_to_end(tmp_path):
    """A PF6 acquisition that plateaus at the -5000 A limit is reconstructed
    by the mapper via the generic primitive."""
    shot = 45965
    n = 1200
    # PF6 gain is -1.0e3, and the baseline is the mean of the record, so a
    # raw interior plateau of amplitude A centred to A*(1-f) maps to
    # -1000*A*(1-f); choose A so the plateau lands on the -5000 A limit.
    plateau = np.zeros(n, dtype=float)
    plateau[400:700] = 1.0
    fraction = plateau.mean()
    amplitude = 5.0 / (1.0 - fraction)
    pf6_raw = amplitude * plateau
    other = np.sin(np.linspace(0.0, 20.0, n)) + np.linspace(0.0, 0.2, n)

    source = tmp_path / "raw.json.gz"
    _write_raw_dump(
        source, shot, {5: other, 59: other, 62: pf6_raw, 65: other}
    )

    _time_axis, currents = vfit_pf(shot, raw_source=source)
    pf6 = currents[PF6_INDEX]

    # The plateau interior was at the clip level before repair and must no
    # longer sit on it afterwards.
    interior = slice(450, 650)
    assert np.min(np.abs(pf6[interior] - (-5000.0))) > 10.0
    assert np.all(np.isfinite(pf6))


@pytest.mark.parametrize(
    ("shot", "repair_enabled"),
    [
        (19000, False),
        (20259, False),
        (38109, False),
        (38110, True),
        (38361, True),
        (45965, True),
        (48372, True),
    ],
)
def test_pf6_saturation_repair_is_scoped_to_the_negative_gain_eras(shot, repair_enabled):
    """The donor (`vest_pf.m`) repairs PF6 unconditionally, but PF6's gain
    flips sign at shot 38110 (+1e3 -> -1e3). Before that the acquisition rail
    appears at *+5000* A, so a -5000 A sample is ordinary measured data and
    "repairing" it would corrupt real signal."""
    from vaft.machine_mapping.utils import resolve_vest_diagnostic as _resolve

    processing = _resolve(shot, "pf_active")["processing"]
    repair = processing.get("saturation_repair")
    gain = float(processing["coil_gains"][5])

    assert bool(repair) is repair_enabled
    # The policy is enabled exactly where the rail is reachable at -5000 A.
    assert (gain < 0) is repair_enabled
    if repair_enabled:
        assert float(repair[5]["value"]) == pytest.approx(-5000.0)


def test_old_shot_pf6_current_near_minus_5000_is_left_alone(tmp_path):
    """A pre-38110 shot whose PF6 legitimately passes -5000 A must not be
    silently rewritten by the repair."""
    shot = 20259  # positive PF6 gain era
    n = 1200
    plateau = np.zeros(n, dtype=float)
    plateau[400:700] = 1.0
    amplitude = -5.0 / (1.0 - plateau.mean())
    pf6_raw = amplitude * plateau
    other = np.sin(np.linspace(0.0, 20.0, n)) + np.linspace(0.0, 0.2, n)

    source = tmp_path / "raw.json.gz"
    _write_raw_dump(source, shot, {5: other, 59: other, 62: pf6_raw, 65: other})

    _time_axis, currents = vfit_pf(shot, raw_source=source)
    pf6 = currents[PF6_INDEX]

    # It really does reach the rail value, and it is preserved as measured.
    assert np.min(np.abs(pf6 - (-5000.0))) < 10.0
