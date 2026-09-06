"""``is_signal_active`` decides by the trace's own level (issue #463).

The variance test used to divide the variance by itself, so it was 1 for any
real signal and the ``var_ratio_thresh`` parameter never governed a verdict.
Both ratios are now relative to the mean absolute level, and these tests pin
the three cases the issue reproduced plus the parameter's newly live branch.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from omas import ODS

import vaft.omas
from vaft.omas.general import classify_shot
from vaft.process import is_signal_active

from _plasma_timing_fixtures import classified_shot, grid, light, pipeline_ods, synthetic_ods


def _rng():
    return np.random.default_rng(463)


def test_a_flat_trace_with_tiny_noise_is_inactive():
    trace = 1.0 + 1e-6 * _rng().standard_normal(1000)
    assert is_signal_active(trace) is False


def test_white_noise_is_active():
    assert is_signal_active(_rng().standard_normal(1000)) is True


def test_a_ramp_is_active():
    assert is_signal_active(np.linspace(0.0, 1.0, 1000)) is True


def test_the_variance_threshold_is_no_longer_dead():
    # A slow, large-amplitude oscillation: the sample-to-sample change is
    # negligible, so only the variance ratio can call it active.
    time = np.linspace(0.0, 1.0, 200_000)
    trace = 5.0 + 2.0 * np.sin(2.0 * np.pi * time)
    assert is_signal_active(trace) is True
    assert is_signal_active(trace, var_ratio_thresh=0.5) is False


def test_an_all_zero_trace_is_inactive_and_does_not_divide_by_zero():
    with np.errstate(all="raise"):
        assert is_signal_active(np.zeros(100)) is False


@pytest.mark.parametrize("trace", [[], [1.0]])
def test_fewer_than_two_samples_is_never_active(trace):
    assert is_signal_active(np.asarray(trace)) is False


def test_the_verdict_does_not_depend_on_units():
    trace = np.linspace(0.0, 1.0, 500) + 0.05 * _rng().standard_normal(500)
    assert is_signal_active(trace) == is_signal_active(1e6 * trace) == is_signal_active(1e-6 * trace)


_shot = classified_shot


@pytest.mark.parametrize(
    "pressure_active, halpha_active, ip_pulse, label, decided_by",
    [
        (False, False, False, "Vacuum", "none"),
        (True, False, False, "BD failure", "pressure_response"),
        (False, True, False, "BD failure", "optical_window"),
        (True, True, True, "Plasma", "ip_pulse"),
        (False, False, True, "Plasma", "ip_pulse"),
    ],
)
def test_classify_shot_decides_from_the_shared_timing(capsys, pressure_active, halpha_active, ip_pulse, label, decided_by):
    """The classifier used to pass ``threshold=`` to a function without that
    argument; the bare ``except`` printed the TypeError and answered
    ``'Vacuum'`` for every shot.  It now reads the plasma timing and the gas
    response and says which check decided."""
    from vaft.omas.shot_class import shot_class

    ods = _shot(pressure_active, halpha_active, ip_pulse)
    record = shot_class(ods)

    assert classify_shot(ods) == label == str(record) == record.label
    assert record.decided_by == decided_by
    assert record.ip_pulse is ip_pulse and record.optical_window is halpha_active
    assert record.pressure_active is pressure_active
    assert "Error in find_shotclass" not in capsys.readouterr().out
    json.dumps(record.record())


def test_a_product_without_barometry_is_judged_on_the_light_alone():
    from vaft.omas.shot_class import shot_class

    vacuum = shot_class(_shot(False, False, False, barometry=False))
    assert vacuum.label == "Vacuum" and vacuum.pressure_active is None
    assert "barometry_absent" in vacuum.flags and "barometry absent" in vacuum.reason
    failed = shot_class(_shot(False, True, False, barometry=False))
    assert failed.label == "BD failure" and failed.decided_by == "optical_window"


@pytest.mark.parametrize("shot", [39915, 41524, 41672])
def test_the_packaged_shots_are_plasmas_by_their_current_pulse(shot):
    from vaft.omas.shot_class import shot_class

    record = shot_class(pipeline_ods(shot))
    assert record.label == "Plasma" and record.decided_by == "ip_pulse"
    assert record.pressure_active is True and record.optical_window is True
    assert classify_shot(vaft.omas.sample_ods()) == "Plasma"


def test_a_product_without_plasma_current_cannot_be_classified():
    from vaft.omas.plasma_timing import PlasmaTimingError

    t = grid()
    with pytest.raises(PlasmaTimingError):
        classify_shot(synthetic_ods(slow=light(t), t=t))


def test_the_lenient_form_agrees_and_answers_none_when_it_cannot_classify():
    ods = _shot(True, True, True)
    assert vaft.omas.find_shotclass(ods) == classify_shot(ods) == "Plasma"
    with pytest.warns(DeprecationWarning, match="halpha_threshold is ignored"):
        assert classify_shot(ods, halpha_threshold=0.5) == "Plasma"
    t = grid()
    assert vaft.omas.find_shotclass(synthetic_ods(slow=light(t), t=t)) is None   # no plasma current at all


# ---------------------------------------------------------------------------
# Review findings on PR #535
# ---------------------------------------------------------------------------


def test_a_nan_in_the_pressure_record_does_not_read_as_a_puff():
    from vaft.omas.shot_class import pressure_response, shot_class

    ods = _shot(False, False, False)
    flat = np.asarray(ods["barometry.gauge.0.pressure.data"], dtype=float).copy()
    flat[10] = np.nan
    ods["barometry.gauge.0.pressure.data"] = flat
    assert pressure_response(ods) is False
    assert shot_class(ods).label == "Vacuum"
    ods["barometry.gauge.0.pressure.data"] = np.full(flat.size, np.nan)
    assert pressure_response(ods) is None


def test_an_unusable_current_lets_the_light_stand_in():
    """A condemned current channel is not a missing plasma: a light window still says Plasma."""
    from vaft.omas.shot_class import shot_class

    ods = _shot(True, True, True)
    ods["magnetics.ip.0.validity"] = -2
    record = shot_class(ods)
    assert record.label == "Plasma" and record.decided_by == "optical_window"
    assert "ip_unusable" in record.flags and not record.ip_pulse
    assert record.summary()["agreement"] == "halpha_only"

    flash = shot_class(_shot(False, True, False))      # usable current, no pulse, light: a flash
    assert flash.label == "BD failure" and flash.decided_by == "optical_window"
