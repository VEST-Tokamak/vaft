"""Shot classification from the shared plasma timing and the gas response.

`classify_shot` once called `is_signal_active(data, threshold=...)` against a
signature without that keyword; a bare `except` swallowed the TypeError and
every shot -- including packaged 39915, an 80 kA discharge -- came back
'Vacuum'.  It now reads the shared timing (`vaft.omas.plasma_timing`) and the
barometry, through `vaft.omas.shot_class`, and says which check decided.
"""

from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np
import pytest

import vaft
import vaft.omas
from vaft.omas.plasma_timing import PlasmaTimingError
from vaft.omas.shot_class import shot_class

from _plasma_timing_fixtures import classified_shot, grid, light, synthetic_ods


PACKAGED_PLASMA_SHOTS = (39915, 41524, 41672)


def _load(shot: int):
    with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return vaft.omas.load(vaft.data.sample(shot, representation="omas"))


@pytest.mark.parametrize("shot", PACKAGED_PLASMA_SHOTS)
def test_a_packaged_discharge_is_not_reported_as_a_vacuum_shot(shot):
    ods = _load(shot)
    assert float(np.max(ods["magnetics.ip.0.data"])) > 1.0e4  # it is a discharge
    assert vaft.omas.classify_shot(ods) == "Plasma"
    assert shot_class(ods).decided_by == "ip_pulse"


@pytest.mark.parametrize("shot", PACKAGED_PLASMA_SHOTS)
def test_the_two_classifiers_agree(shot):
    """They used to disagree: 'Vacuum' from one, 'Plasma' from the other."""
    ods = _load(shot)
    assert vaft.omas.find_shotclass(ods) == vaft.omas.classify_shot(ods)


def test_each_class_is_reachable():
    assert vaft.omas.classify_shot(classified_shot(False, False, False)) == "Vacuum"
    assert vaft.omas.classify_shot(classified_shot(True, False, False)) == "BD failure"
    assert vaft.omas.classify_shot(classified_shot(True, True, True)) == "Plasma"
    # Gas and light, but the current never rose: the breakdown failed.
    assert vaft.omas.classify_shot(classified_shot(True, True, False)) == "BD failure"
    # A current pulse is a plasma whatever the gauge says.
    assert vaft.omas.classify_shot(classified_shot(False, False, True)) == "Plasma"


def test_a_missing_current_is_reported_not_called_vacuum():
    """A product without the plasma current cannot be classified either way, and
    asking must not create the path."""
    t = grid()
    ods = synthetic_ods(slow=light(t), t=t)
    before = sorted(map(str, ods.flat().keys()))
    with pytest.raises(PlasmaTimingError, match="magnetics.ip.0"):
        vaft.omas.classify_shot(ods)
    assert sorted(map(str, ods.flat().keys())) == before
    assert vaft.omas.find_shotclass(ods) is None  # the lenient form still answers


def test_a_missing_gauge_is_judged_on_the_light_alone():
    """An unrecorded gauge is not a vacuum shot: the record says the check was skipped."""
    record = shot_class(classified_shot(False, True, False, barometry=False))
    assert record.label == "BD failure" and record.decided_by == "optical_window"
    assert record.pressure_active is None and "barometry_absent" in record.flags


def test_the_pressure_threshold_reaches_the_detector():
    """A faint gauge response between two thresholds: active at 0.01, flat at 0.05."""
    ods = classified_shot(False, False, False)
    faint = np.asarray(ods["barometry.gauge.0.pressure.data"], dtype=float).copy()
    faint[::2] += 0.02
    ods["barometry.gauge.0.pressure.data"] = faint
    assert vaft.omas.classify_shot(ods) == "BD failure"
    assert vaft.omas.classify_shot(ods, pressure_threshold=0.05) == "Vacuum"
    # the light is judged by the timing, not by a threshold: the old keyword only warns
    with pytest.warns(DeprecationWarning, match="halpha_threshold is ignored"):
        assert vaft.omas.classify_shot(ods, halpha_threshold=0.05) == "BD failure"
